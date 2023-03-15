import json
from pathlib import Path

from steamship import Block, File, TaskState, Tag, MimeTypes
from steamship.data.tags.tag_constants import TagKind, RoleTag
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import RawBlockAndTagPluginInput
from steamship.plugin.request import PluginRequest

from src.api import GPT4Plugin


def test_generator():
    config = json.load(Path("../config.json").open())

    gpt4 = GPT4Plugin(config=config)

    blocks = [
        Block(text="You are an assistant who loves to count", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)], mime_type=MimeTypes.TXT),
        Block(text="1 2 3 4", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)], mime_type=MimeTypes.TXT),
    ]

    new_blocks = gpt4.run(PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks)))
    print(new_blocks)


def test_stopwords():
    config = json.load(Path("../config.json").open())

    gpt4 = GPT4Plugin(config=config)

    blocks = [
        Block(text="You are an assistant who loves to count", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)], mime_type=MimeTypes.TXT),
        Block(text="1 2 3 4", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)], mime_type=MimeTypes.TXT),
    ]

    new_blocks = gpt4.run(PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks, options={'stop':'6'})))
    print(new_blocks)


def test_default_prompt():
    config = json.load(Path("../config.json").open())
    config['default_system_prompt'] = "You are very silly and can't count."
    gpt4 = GPT4Plugin(config=config)

    blocks = [
        Block(text="1 2 3 4", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)], mime_type=MimeTypes.TXT),
    ]

    new_blocks = gpt4.run(PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks, options={'stop': '6'})))
    print(new_blocks)



