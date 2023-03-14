import json
from pathlib import Path

from steamship import Block, File, TaskState
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import RawBlockAndTagPluginInput
from steamship.plugin.request import PluginRequest

from src.api import GPT4Plugin


def test_generator():
    config = json.load(Path("../config.json").open())

    gpt4 = GPT4Plugin(config=config)
    request = PluginRequest(data=RawBlockAndTagPluginInput(blocks=[Block(text="a cat riding a bicycle")]))
    response = dalle.run(request)

    assert response.status.state is TaskState.succeeded
    assert response.data is not None

    assert len(response.data.blocks) == 1
    assert response.data.blocks[0].url is not None
    print(response.data.blocks[0].url)



