import logging
from typing import Any, Dict, List, Optional, Type
import json
import openai
from pydantic import Field
from steamship import Steamship, Block, Tag, SteamshipError
from steamship.data.tags.tag_constants import TagKind, RoleTag
from steamship.invocable import Config, InvocableResponse, InvocationContext
from steamship.plugin.generator import Generator
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import RawBlockAndTagPluginInput
from steamship.plugin.outputs.raw_block_and_tag_plugin_output import RawBlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    before_sleep_log, wait_exponential_jitter,
)



class GPT4Plugin(Generator):
    """
    Plugin for generating images from text prompts from DALL-E.
    """

    class GPT4PluginConfig(Config):
        openai_api_key: str = Field("",
                                    description="An openAI API key to use. If left default, will use Steamship's API key.")
        max_tokens: int = Field(256, description="The maximum number of tokens to generate per request. Can be overridden in runtime options.")
        model: Optional[str] = Field("gpt-4",
                                     description="The OpenAI model to use.  Can be a pre-existing fine-tuned model.")
        temperature: Optional[float] = Field(0.4,
                                             description="Controls randomness. Lower values produce higher likelihood / more predictable results; higher values produce more variety. Values between 0-1.")
        top_p: Optional[int] = Field(1,
                                     description="Controls the nucleus sampling, where the model considers the results of the tokens with top_p probability mass. Values between 0-1.")
        presence_penalty: Optional[int] = Field(0,
                                                description="Control how likely the model will reuse words. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Number between -2.0 and 2.0.")
        frequency_penalty: Optional[int] = Field(0,
                                                 description="Control how likely the model will reuse words. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Number between -2.0 and 2.0.")
        moderate_output: bool = Field(True,
                                      description="Pass the generated output back through OpenAI's moderation endpoint and throw an exception if flagged.")
        max_retries: int = Field(8, description="Maximum number of retries to make when generating.")
        request_timeout: Optional[float] = Field(600,
                                                 description="Timeout for requests to OpenAI completion API. Default is 600 seconds.")
        n: Optional[int] = Field(1, description="How many completions to generate for each prompt.")

        default_role: str = Field(RoleTag.USER.value, description="The default role to use for a block that does not have a Tag of kind='role'")

        default_system_prompt: str = Field("", description="System prompt that will be prepended before every request")



    @classmethod
    def config_cls(cls) -> Type[Config]:
        return cls.GPT4PluginConfig

    config: GPT4PluginConfig

    def __init__(
            self,
            client: Steamship = None,
            config: Dict[str, Any] = None,
            context: InvocationContext = None,
    ):
        super().__init__(client, config, context)
        openai.api_key = self.config.openai_api_key



    def prepare_message(self, block: Block) -> Dict[str, str]:
        role = None
        for tag in block.tags:
            if tag.kind == TagKind.ROLE:
                role = tag.name

        if role is None:
            role = self.config.default_role

        return {
            "role" : role,
            "content" : block.text
        }


    def prepare_messages(self, blocks: List[Block]) -> List[Dict[str, str]]:
        messages = []
        if self.config.default_system_prompt != "":
            messages.append({
            "role" : RoleTag.SYSTEM,
            "content" : self.config.default_system_prompt
        })
        # TODO: remove is_text check here when can handle image etc. input
        messages.extend([self.prepare_message(block) for block in blocks if block.text is not None and block.text != ""])
        return messages

    def generate_with_retry(self, user: str, messages: List[Dict[str, str]], options: Dict) -> List[Block]:
        """Use tenacity to retry the completion call."""

        logging.info(f"Making OpenAI dall-e call on behalf of user with id: {user}")
        """Call the API to generate the next section of text."""

        stopwords = options.get('stop', None) if options is not None else None

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(jitter=5),
            before_sleep=before_sleep_log(logging.root, logging.INFO),
            retry=(
                    retry_if_exception_type(openai.error.Timeout)
                    | retry_if_exception_type(openai.error.APIError)
                    | retry_if_exception_type(openai.error.APIConnectionError)
                    | retry_if_exception_type(openai.error.RateLimitError)
            ),
            after=after_log(logging.root, logging.INFO),
        )
        def _generate_with_retry() -> Any:
            return openai.ChatCompletion.create(
                model=self.config.model,
                messages=messages,
                user=user,
                presence_penalty=self.config.presence_penalty,
                frequency_penalty=self.config.frequency_penalty,
                max_tokens=self.config.max_tokens,
                stop=stopwords,
                n=self.config.n,
            )

        openai_result = _generate_with_retry()
        logging.info("Retry statistics: " + json.dumps(_generate_with_retry.retry.statistics))

        # Fetch text from responses
        texts = [choice['message']['content'] for choice in openai_result['choices']]


        return [Block(text = text, tags=[Tag(kind=TagKind.ROLE, name=RoleTag.ASSISTANT)]) for text in texts]



    def run(
            self, request: PluginRequest[RawBlockAndTagPluginInput]
    ) -> InvocableResponse[RawBlockAndTagPluginOutput]:
        """Run the image generator against all the text, combined """

        self.config.extend_with_dict(request.data.options, overwrite=True)

        messages = self.prepare_messages(request.data.blocks)
        user_id = self.context.user_id if self.context is not None else "testing"
        generated_blocks = self.generate_with_retry(messages=messages, user=user_id, options=request.data.options)

        return InvocableResponse(data=RawBlockAndTagPluginOutput(blocks= generated_blocks))
