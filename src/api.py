import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Type

import litellm
import openai
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
    wait_exponential_jitter, retry_if_exception,
)

BILLING_LOCK_TIMEOUT_SECONDS = 10.0
# This is a massive multiplier, but we are going to handle the fractional cents pricing-side.
# This is basically figured by going into the pricing data that I had for LiteLLM and finding the
# maximum amount of significant figures for any costing and making sure we can capture that precision
# in an int.
PRICING_UNITS_MULTIPLIER = 10_000_000_000_000

PLUGIN_HANDLED_KEYS = {"stream", "user", "messages"}
DEFAULT_MODEL = "gpt-4-0613"


class BillingCallback:
    def __init__(self):
        self.cost = None
        self.completion_response = None
        self.lock = threading.Lock()
        assert self.lock.acquire(blocking=False), "Could not acquire billing lock"

    def __call__(self, kwargs, completion_response, start_time, end_time):
        if "complete_streaming_response" in kwargs:
            completion_response = kwargs["complete_streaming_response"]
            self.cost = litellm.completion_cost(completion_response)
            self.completion_response = completion_response
            self.lock.release()

    def usage(self, completion_id: str) -> [UsageReport]:
        assert self.lock.acquire(timeout=BILLING_LOCK_TIMEOUT_SECONDS), "Billing calculation timed out"
        return [
            UsageReport(
                operation_type=OperationType.RUN,
                operation_unit=OperationUnit.UNITS,
                operation_amount=int(self.cost * PRICING_UNITS_MULTIPLIER),
                audit_id=completion_id
            )
        ]


class NoopBilling(BillingCallback):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def usage(self, completion_id: str) -> [UsageReport]:
        return []


class LiteLLMPlugin(StreamingGenerator):
    """
    Plugin for generating text using LiteLLM-supported models.
    """

    class LiteLLMPluginConfig(Config):
        litellm_env: str = Field(
            "",
            description="LiteLLM provider environment, in the form <key>:<value>[;<key>:<value>...].  See LiteLLM "
                        "documentation for supported variables.  Params must end with _API_KEY, and in the case of "
                        "'AZURE', _API_VERSION, or _API_BASE.  Leaving this empty uses Steamship's environment and "
                        "thus Steamship billing."
        )
        max_retries: int = Field(
            8, description="Maximum number of retries to make when generating."
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
        # Load original env before it is read from TOML, so we know whether to bill usage or not.
        original_env = config.get("litellm_env", "")
        super().__init__(client, config, context)
        if config.get("n", 1) != 1:
            raise SteamshipError("There is currently a known bug with this plugin and config value n != 1.")

        self.apply_env(self.config.litellm_env)
        self.steamship_billing = (original_env == "")

    @staticmethod
    def get_envs(env_config: str) -> {str: str}:
        envs = {}
        for kv_pair in env_config.split(";"):
            kv = kv_pair.split(":")
            if len(kv) != 2:
                raise SteamshipError("litellm_env must be of the form <key>:<value>[;<key>:<value>...]")
            k, v = kv
            if not k.endswith("_API_KEY") or k.endswith("_API_BASE") or k.endswith("_API_VERSION"):
                raise SteamshipError("litellm environment keys must end with _API_KEY, _API_BASE, or _API_VERSION")
            envs[k] = v
        return envs

    @staticmethod
    def apply_env(env_config: str):
        for k, v in LiteLLMPlugin.get_envs(env_config).items():
            os.environ[k] = v

    def prepare_message(self, block: Block, options: Dict[str, Any]) -> Optional[Dict[str, str]]:
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
            role = options.get("default_role", RoleTag.USER.value)

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

    def prepare_messages(self, blocks: List[Block], options: Dict[str, Any]) -> List[Dict[str, str]]:
        messages = []
        default_system_prompt = options.get("default_system_prompt")
        if default_system_prompt:
            messages.append(
                {"role": RoleTag.SYSTEM, "content": default_system_prompt}
            )
        # TODO: remove is_text check here when can handle image etc. input
        messages.extend(
            [
                msg for msg in
                (self.prepare_message(block, options) for block in blocks if block.text is not None and block.text != "")
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
        logging.info(
            f"Making LiteLLM call on behalf of user with id: {user}"
        )
        options = options or {}
        if "model" not in options:
            options["model"] = DEFAULT_MODEL
        model = options["model"]
        functions = options.get("functions", None)

        billing_callback = BillingCallback() if self.steamship_billing else NoopBilling()

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(jitter=5),
            before_sleep=before_sleep_log(logging.root, logging.INFO),
            retry=(
                retry_if_exception_type(openai.APITimeoutError)
                | retry_if_exception(
                    lambda e: isinstance(e, openai.APIError) and not
                    ("does not support parameters" in e.message) and not
                    (isinstance(e, openai.AuthenticationError)))
                | retry_if_exception_type(openai.APIConnectionError)
                | retry_if_exception_type(openai.RateLimitError)
                | retry_if_exception_type(
                    ConnectionError
                )  # handle 104s that manifest as ConnectionResetError
            ),
            after=after_log(logging.root, logging.INFO),
        )
        def _generate_with_retry() -> Any:
            kwargs = dict(
                **options,
                messages=messages,
                user=user,
                stream=True,
            )

            _, provider, _, _ = litellm.get_llm_provider(model)
            if provider == "replicate":
                # TODO This probably can be sidestepped by just using options for this.
                del kwargs["presence_penalty"]
                del kwargs["frequency_penalty"]
                del kwargs["user"]

            logging.info("calling litellm.completion",
                         extra={"messages": messages, "functions": functions})
            return litellm.completion(**kwargs)

        litellm.success_callback = [billing_callback]
        litellm_result = _generate_with_retry()
        logging.info(
            "Retry statistics: " + json.dumps(_generate_with_retry.retry.statistics)
        )

        output_texts = [  # Collect output text for usage counting
            "" for _ in range(options.get('n', 1))
        ]
        # iterate through the stream of events
        completion_id = ""
        returned_function_parts = [  # did each result return a function
            [] for _ in range(options.get('n', 1))
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

        return billing_callback.usage(completion_id)

    def _reassemble_function_call(self, function_call_chunks: List[dict]) -> dict:
        result = {}
        for chunk in function_call_chunks:
            model = chunk.model_dump()
            for key in model.keys():
                if key not in result:
                    result[key] = ""
                if model[key]:
                    result[key] += model[key]
        return result

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
        return moderation.results[0].flagged

    def _validate_options(self, options: Dict[str, Any]):
        if not options:
            return
        invalid_keys = PLUGIN_HANDLED_KEYS - options.keys()
        if invalid_keys:
            raise SteamshipError("The following keys are handled entirely by the plugin and may not be provided as "
                                 "options: " + (', '.join(invalid_keys)))
        if 'litellm_env' in options:
            raise SteamshipError("Configured environment (litellm_env) may not be overridden in options")
        if options.get('n', 1) > 1:
            raise SteamshipError("There is currently a known bug in implementation with this plugin where n > 1.")

    def run(
        self, request: PluginRequest[RawBlockAndTagPluginInputWithPreallocatedBlocks]
    ) -> InvocableResponse[StreamCompletePluginOutput]:
        """Run the text generator against all the text, combined"""

        options = request.data.options
        self._validate_options(options)
        messages = self.prepare_messages(request.data.blocks, options)
        if options.get("moderate_output", False) and self._flagged(messages):
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

        options = request.data.options
        self._validate_options(options)

        # We return one block per completion-choice we're configured for (config.n)
        # This expectation is coded into the generate_with_retry method
        block_types_to_create = [MimeTypes.TXT.value] * options.get("n", 1)
        return InvocableResponse(
            data=BlockTypePluginOutput(block_types_to_create=block_types_to_create)
        )
