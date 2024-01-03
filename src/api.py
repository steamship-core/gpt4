import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Type

import litellm
import openai
import tiktoken
from pydantic import Field
from steamship import Steamship, Block, Tag, SteamshipError, MimeTypes
from steamship.data.tags.tag_constants import TagKind, RoleTag, TagValueKey, ChatTag
from steamship.invocable import Config, InvocableResponse, InvocationContext
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import (
    RawBlockAndTagPluginInput,
)
from steamship.plugin.inputs.raw_block_and_tag_plugin_input_with_preallocated_blocks import (
    RawBlockAndTagPluginInputWithPreallocatedBlocks,
)
from steamship.plugin.outputs.block_type_plugin_output import BlockTypePluginOutput
from steamship.plugin.outputs.plugin_output import (
    UsageReport,
    OperationType,
    OperationUnit,
)
from steamship.plugin.outputs.stream_complete_plugin_output import (
    StreamCompletePluginOutput,
)
from steamship.plugin.request import PluginRequest
from steamship.plugin.streaming_generator import StreamingGenerator
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    before_sleep_log,
    wait_exponential_jitter,
)

VALID_MODELS_FOR_BILLING = [
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4-1106-preview"
]


class LiteLLMPlugin(StreamingGenerator):
    """
    Plugin for generating text using LiteLLM-supported models.
    """

    class LiteLLMPluginConfig(Config):
        openai_api_key: str = Field(
            "",
            description="An openAI API key to use. If left default, will use Steamship's API key.",
        )
        max_tokens: int = Field(
            256,
            description="The maximum number of tokens to generate per request. Can be overridden in runtime options.",
        )
        model: Optional[str] = Field(
            "gpt-4-0613",
            description="The model name to use. Can be a pre-existing fine-tuned model.",
        )
        temperature: Optional[float] = Field(
            0.4,
            description="Controls randomness. Lower values produce higher likelihood / more predictable results; "
            "higher values produce more variety. Values between 0-1.",
        )
        top_p: Optional[int] = Field(
            1,
            description="Controls the nucleus sampling, where the model considers the results of the tokens with "
            "top_p probability mass. Values between 0-1.",
        )
        presence_penalty: Optional[int] = Field(
            0,
            description="Control how likely the model will reuse words. Positive values penalize new tokens based on "
            "whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Number between -2.0 and 2.0.",
        )
        frequency_penalty: Optional[int] = Field(
            0,
            description="Control how likely the model will reuse words. Positive values penalize new tokens based on "
            "their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Number between -2.0 and 2.0.",
        )
        moderate_output: bool = Field(
            True,
            description="Pass the generated output back through OpenAI's moderation endpoint and throw an exception "
            "if flagged.",
        )
        max_retries: int = Field(
            8, description="Maximum number of retries to make when generating."
        )
        request_timeout: Optional[float] = Field(
            600,
            description="Timeout in seconds for requests to the completion API. Default is 600 seconds.",
        )
        n: Optional[int] = Field(
            1, description="How many completions to generate for each prompt."
        )
        default_role: str = Field(
            RoleTag.USER.value,
            description="The default role to use for a block that does not have a Tag of kind='role'",
        )
        default_system_prompt: str = Field(
            "", description="System prompt that will be prepended before every request"
        )

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return cls.LiteLLMPluginConfig

    config: LiteLLMPluginConfig

    def __init__(
        self,
        client: Steamship = None,
        config: Dict[str, Any] = None,
        context: InvocationContext = None,
    ):
        # Load original api key before it is read from TOML, so we know to restrict models for billing
        original_api_key = config.get("openai_api_key", "")
        super().__init__(client, config, context)
        if original_api_key == "" and self.config.model not in VALID_MODELS_FOR_BILLING:
            raise SteamshipError(
                f"This plugin cannot be used with model {self.config.model} while using Steamship's API key. Valid models are {VALID_MODELS_FOR_BILLING}"
            )
        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key

    def prepare_message(self, block: Block) -> Optional[Dict[str, str]]:
        role = None
        name = None
        function_selection = False

        for tag in block.tags:
            if tag.kind == TagKind.ROLE:
                # this is a legacy way of extracting roles
                role = tag.name

            if tag.kind == TagKind.CHAT and tag.name == ChatTag.ROLE:
                if values := tag.value:
                    if role_name := values.get(TagValueKey.STRING_VALUE, None):
                        role = role_name

            if tag.kind == TagKind.ROLE and tag.name == RoleTag.FUNCTION:
                if values := tag.value:
                    if fn_name := values.get(TagValueKey.STRING_VALUE, None):
                        name = fn_name

            if tag.kind == "function-selection":
                function_selection = True

            if tag.kind == "name":
                name = tag.name

        if role is None:
            role = self.config.default_role

        if role not in ["function", "system", "assistant", "user"]:
            logging.warning(f"unsupported role {role} found in message. skipping...")
            return None

        if role == "function" and not name:
            name = "unknown"  # protect against missing function names

        if function_selection:
            return {"role": RoleTag.ASSISTANT.value, "content": None, "function_call": json.loads(block.text)}

        if name:
            return {"role": role, "content": block.text, "name": name}

        return {"role": role, "content": block.text}

    def prepare_messages(self, blocks: List[Block]) -> List[Dict[str, str]]:
        messages = []
        if self.config.default_system_prompt != "":
            messages.append(
                {"role": RoleTag.SYSTEM, "content": self.config.default_system_prompt}
            )
        # TODO: remove is_text check here when can handle image etc. input
        messages.extend(
            [
                msg for msg in
                (self.prepare_message(block) for block in blocks if block.text is not None and block.text != "")
                if msg is not None
            ]
        )
        return messages

    def generate_with_retry(
        self,
        user: str,
        messages: List[Dict[str, str]],
        options: Dict,
        output_blocks: List[Block],
    ) -> List[UsageReport]:
        """Call the API to generate the next section of text."""
        # TODO Fix
        logging.info(
            f"Making LiteLLM call on behalf of user with id: {user}"
        )
        options = options or {}
        stopwords = options.get("stop", None)
        functions = options.get("functions", None)

        # record the time before the request is sent
        start_time = time.time()

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
                | retry_if_exception_type(
                    ConnectionError
                )  # handle 104s that manifest as ConnectionResetError
            ),
            after=after_log(logging.root, logging.INFO),
        )
        def _generate_with_retry() -> Any:
            kwargs = dict(
                model=self.config.model,
                messages=messages,
                user=user,
                presence_penalty=self.config.presence_penalty,
                frequency_penalty=self.config.frequency_penalty,
                max_tokens=self.config.max_tokens,
                stop=stopwords,
                n=self.config.n,
                temperature=self.config.temperature,
                stream=True,
            )
            if functions:
                kwargs = {**kwargs, "functions": functions}

            logging.info("calling litellm.completion",
                         extra={"messages": messages, "functions": functions})
            return litellm.completion(**kwargs)

        litellm_result = _generate_with_retry()
        logging.info(
            "Retry statistics: " + json.dumps(_generate_with_retry.retry.statistics)
        )

        output_texts = [  # Collect output text for usage counting
            "" for _ in range(self.config.n)
        ]
        # iterate through the stream of events
        completion_id = ""
        returned_function_parts = [  # did each result return a function
            [] for _ in range(self.config.n)
        ]
        for chunk in litellm_result:
            if id := chunk.get("id"):
                completion_id = id
            for chunk_choice in chunk["choices"]:
                chunk_message = chunk_choice["delta"]  # extract the message
                block_index = chunk_choice["index"]
                output_block = output_blocks[block_index]
                if role := chunk_message.get("role"):
                    Tag.create(
                        self.client,
                        file_id=output_block.file_id,
                        block_id=output_block.id,
                        kind=TagKind.ROLE,
                        name=RoleTag(role),
                    )
                if text_chunk := chunk_message.get("content"):
                    output_block.append_stream(bytes(text_chunk, encoding="utf-8"))
                    output_texts[block_index] += text_chunk
                if function_call := chunk_message.get("function_call"):
                    returned_function_parts[block_index].append(function_call)

        for i, block_returned_function_parts in enumerate(returned_function_parts):
            if len(block_returned_function_parts) > 0:
                function_call = self._reassemble_function_call(
                    block_returned_function_parts
                )
                output_texts[i] = json.dumps(function_call)
                output_blocks[i].append_stream(
                    bytes(
                        json.dumps({"function_call": function_call}), encoding="utf-8"
                    )
                )

        for output_block in output_blocks:
            output_block.finish_stream()

        usage_reports = self._calculate_usage(messages, output_texts, completion_id)

        return usage_reports

    def _reassemble_function_call(self, function_call_chunks: List[dict]) -> dict:
        result = {}
        for chunk in function_call_chunks:
            for key in chunk.keys():
                if key not in result:
                    result[key] = ""
                result[key] += chunk[key]
        return result

    def _calculate_usage(
        self,
        messages: List[Dict[str, str]],
        output_texts: List[str],
        completion_id: str,
    ) -> [UsageReport]:
        # for token usage tracking, we need to include not just the token usage, but also completion id
        # that will allow proper usage aggregation for n > 1 cases
        encoding = tiktoken.encoding_for_model(self.config.model)
        output_tokens = sum([len(encoding.encode(text)) for text in output_texts])
        prompt_tokens = sum(
            [
                len(encoding.encode(message.get("content", "") or ""))
                for message in messages
            ]
        )
        usage_reports = [
            UsageReport(
                operation_type=OperationType.RUN,
                operation_unit=OperationUnit.PROMPT_TOKENS,
                operation_amount=prompt_tokens,
                audit_id=completion_id,
            ),
            UsageReport(
                operation_type=OperationType.RUN,
                operation_unit=OperationUnit.SAMPLED_TOKENS,
                operation_amount=output_tokens,
                audit_id=completion_id,
            ),
        ]
        return usage_reports

    @staticmethod
    def _flagged(messages: List[Dict[str, str]]) -> bool:
        input_text = "\n\n".join(
            [
                json.dumps(value)
                for role_dict in messages
                for value in role_dict.values()
            ]
        )
        moderation = litellm.moderation(input=input_text)
        return moderation["results"][0]["flagged"]

    def run(
        self, request: PluginRequest[RawBlockAndTagPluginInputWithPreallocatedBlocks]
    ) -> InvocableResponse[StreamCompletePluginOutput]:
        """Run the text generator against all the text, combined"""

        self.config.extend_with_dict(request.data.options, overwrite=True)

        messages = self.prepare_messages(request.data.blocks)
        if self.config.moderate_output and self._flagged(messages):
            # Before we bail, we have to mark the blocks as failed -- otherwise they will remain forever in the `streaming` state
            try:
                if request.data.output_blocks:
                    for block in request.data.output_blocks:
                        block.abort_stream()
            except BaseException as ex:
                raise SteamshipError(
                    "Sorry, this content is flagged as inappropriate. Additionally, we were unable to abort the block stream.",
                    error=ex
                )
            raise SteamshipError(
                "Sorry, this content is flagged as inappropriate."
            )
        user_id = self.context.user_id if self.context is not None else "testing"
        usage_reports = self.generate_with_retry(
            messages=messages,
            user=user_id,
            options=request.data.options,
            output_blocks=request.data.output_blocks,
        )

        return InvocableResponse(data=StreamCompletePluginOutput(usage=usage_reports))

    def determine_output_block_types(
        self, request: PluginRequest[RawBlockAndTagPluginInput]
    ) -> InvocableResponse[BlockTypePluginOutput]:

        if request.data.options is not None and 'model' in request.data.options:
            raise SteamshipError("Model may not be overridden in options")

        self.config.extend_with_dict(request.data.options, overwrite=True)

        # We return one block per completion-choice we're configured for (config.n)
        # This expectation is coded into the generate_with_retry method
        block_types_to_create = [MimeTypes.TXT.value] * self.config.n
        return InvocableResponse(
            data=BlockTypePluginOutput(block_types_to_create=block_types_to_create)
        )
