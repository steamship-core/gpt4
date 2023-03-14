"""Test assemblyai-s2t-blockifier via integration tests."""
import io

from PIL import Image


import pytest
from steamship import Block, File, PluginInstance, Steamship, TaskState, MimeTypes
from steamship.data import GenerationTag, TagKind, TagValueKey

GENERATOR_HANDLE = "dall-e"


@pytest.fixture
def steamship() -> Steamship:
    """Instantiate a Steamship client."""
    return Steamship(profile="test")




def test_generator(steamship: Steamship):


    with steamship.temporary_workspace() as steamship:
        dalle = steamship.use_plugin("dall-e")

        task = dalle.generate(text="A cat on a bicycle", append_output_to_file=True)
        task.wait()
        blocks = task.output.blocks

        assert blocks is not None
        assert len(blocks) == 1
        assert blocks[0].mime_type == MimeTypes.PNG
        content = blocks[0].raw()
        img = Image.open(io.BytesIO(content))
        img.show()



