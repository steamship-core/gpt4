from steamship import Block, Tag, MimeTypes
from steamship.data.tags.tag_constants import TagKind, RoleTag
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import RawBlockAndTagPluginInput
from steamship.plugin.request import PluginRequest

from src.api import GPT4Plugin


def test_generator():
    gpt4 = GPT4Plugin(config={"n": 4})

    blocks = [
        Block(text="You are an assistant who loves to count", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
              mime_type=MimeTypes.TXT),
        Block(text="1 2 3 4", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)], mime_type=MimeTypes.TXT),
    ]

    new_blocks = gpt4.run(PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks)))
    assert len(new_blocks.data.blocks) == 4
    for block in new_blocks.data.blocks:
        assert block.text.strip() == "5 6 7 8 9 10"
        usage_found = False
        for tag in block.tags:
            if tag.kind == "token_usage":
                usage_found = True
        assert usage_found


def test_stopwords():
    gpt4 = GPT4Plugin()

    blocks = [
        Block(text="You are an assistant who loves to count", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
              mime_type=MimeTypes.TXT),
        Block(text="1 2 3 4", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)], mime_type=MimeTypes.TXT),
    ]

    new_blocks = gpt4.run(PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks, options={'stop': '6'})))
    assert len(new_blocks.data.blocks) == 1
    assert new_blocks.data.blocks[0].text.strip() == "5"


def test_default_prompt():
    gpt4 = GPT4Plugin(config={'default_system_prompt': "You are very silly and are afraid of numbers. When you see "
                                                       "them you scream: 'YIKES!'"})

    blocks = [
        Block(text="1 2 3 4", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)], mime_type=MimeTypes.TXT),
    ]

    new_blocks = gpt4.run(PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks, options={'stop': '6'})))
    assert len(new_blocks.data.blocks) == 1
    assert new_blocks.data.blocks[0].text.strip() == "YIKES!"
