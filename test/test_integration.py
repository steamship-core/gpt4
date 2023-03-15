"""Test assemblyai-s2t-blockifier via integration tests."""
import io

from PIL import Image


import pytest
from steamship import Block, File, PluginInstance, Steamship, TaskState, MimeTypes,Tag
from steamship.data.tags.tag_constants import RoleTag, TagKind
from steamship.data import GenerationTag, TagKind, TagValueKey



@pytest.fixture
def steamship() -> Steamship:
    """Instantiate a Steamship client."""
    return Steamship(profile="dave-staging")




def test_generator(steamship: Steamship):


    with steamship.temporary_workspace() as steamship:

        gpt4 = steamship.use_plugin("gpt-4")

        file = File.create(steamship, blocks =
         [
            Block(text="You are an assistant who loves to count", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
                  mime_type=MimeTypes.TXT),
            Block(text="1 2 3 4", tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)], mime_type=MimeTypes.TXT),
        ])

        generate_task = gpt4.generate(input_file_id=file.id)
        generate_task.wait()
        output = generate_task.output





