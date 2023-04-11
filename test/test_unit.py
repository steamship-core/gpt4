from steamship import Block, Tag, MimeTypes, SteamshipError
from steamship.data.tags.tag_constants import TagKind, RoleTag
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import (
    RawBlockAndTagPluginInput,
)
from steamship.plugin.request import PluginRequest
from openai.error import InvalidRequestError

from src.api import GPT4Plugin
import pytest


def test_generator():
    gpt4 = GPT4Plugin(config={"n": 4})

    blocks = [
        Block(
            text="You are an assistant who loves to count",
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
            mime_type=MimeTypes.TXT,
        ),
        Block(
            text="1 2 3 4",
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
            mime_type=MimeTypes.TXT,
        ),
    ]

    new_blocks = gpt4.run(PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks)))
    assert len(new_blocks.data.blocks) == 4
    for block in new_blocks.data.blocks:
        assert block.text.strip().startswith("5 6 7 8 9 10")

    assert new_blocks.data.usage is not None
    assert len(new_blocks.data.usage) == 2


def test_stopwords():
    gpt4 = GPT4Plugin()

    blocks = [
        Block(
            text="You are an assistant who loves to count",
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
            mime_type=MimeTypes.TXT,
        ),
        Block(
            text="1 2 3 4",
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
            mime_type=MimeTypes.TXT,
        ),
    ]

    new_blocks = gpt4.run(
        PluginRequest(
            data=RawBlockAndTagPluginInput(blocks=blocks, options={"stop": "6"})
        )
    )
    assert len(new_blocks.data.blocks) == 1
    assert new_blocks.data.blocks[0].text.strip() == "5"


def test_default_prompt():
    gpt4 = GPT4Plugin(
        config={
            "default_system_prompt": "You are very silly and are afraid of numbers. When you see "
            "them you scream: 'YIKES!'"
        }
    )

    blocks = [
        Block(
            text="1 2 3 4",
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
            mime_type=MimeTypes.TXT,
        ),
    ]

    new_blocks = gpt4.run(
        PluginRequest(
            data=RawBlockAndTagPluginInput(blocks=blocks, options={"stop": "6"})
        )
    )
    assert len(new_blocks.data.blocks) == 1
    assert new_blocks.data.blocks[0].text.strip() == "YIKES!"


def test_flagged_prompt():
    gpt4 = GPT4Plugin()

    blocks = [
        Block(
            text=" <Insert something super offensive here to run this test>",
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
            mime_type=MimeTypes.TXT,
        ),
    ]

    with pytest.raises(SteamshipError):
        new_blocks = gpt4.run(
            PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks))
        )
        assert len(new_blocks.data.blocks) == 1
        assert new_blocks.data.blocks[0].text.strip() == "YIKES!"


def test_too_long_input():
    gpt4 = GPT4Plugin()

    blocks = [
        Block(
            text="token " * 9000,
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
            mime_type=MimeTypes.TXT,
        ),
    ]

    with pytest.raises(InvalidRequestError):
        _ = gpt4.run(
            PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks))
        )


def test_filtered_length():
    gpt4 = GPT4Plugin()

    blocks = [
        Block(
            text="token " * 1000,
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER),
                  Tag(kind="num", name=str(i))],
            mime_type=MimeTypes.TXT,
        ) for i in range(10)
    ]

    # Expect 7 blocks, not 8, because each gets measured as 1001 since it ends with a space
    assert len(gpt4.filter_blocks_for_prompt_length(blocks)) == 7

def test_auto_filter_length():
    gpt4 = GPT4Plugin(config={"auto_limit_prompt_length": True})

    blocks = [
        Block(
            text="token " * 1000,
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER),
                  Tag(kind="num", name=str(i))],
            mime_type=MimeTypes.TXT,
        ) for i in range(10)
    ]

    new_blocks = gpt4.run(
        PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks))
    )
    assert len(new_blocks.data.blocks) == 1
    assert "token" in new_blocks.data.blocks[0].text.strip()