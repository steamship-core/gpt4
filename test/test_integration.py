"""Test gpt-4 generation via integration tests."""
from steamship import Block, File, Steamship, MimeTypes, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import RoleTag

GENERATOR_HANDLE = "gpt-4-enias"
PROFILE = "staging"


def test_generator():
    with Steamship.temporary_workspace(profile=PROFILE) as steamship:
        gpt4 = steamship.use_plugin(GENERATOR_HANDLE)
        print(gpt4)
        file = File.create(
            steamship,
            blocks=[
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
            ],
        )

        generate_task = gpt4.generate(input_file_id=file.id)
        generate_task.wait()
        output = generate_task.output
        assert len(output.blocks) == 1
        for block in output.blocks:
            assert block.text.strip().startswith("5")
            assert block.text.strip() == "5 6 7 8 9 10"


def test_generator_without_role():
    with Steamship.temporary_workspace() as steamship:
        gpt4 = steamship.use_plugin(GENERATOR_HANDLE)
        file = File.create(
            steamship,
            blocks=[
                Block(text="1 2 3 4"),
            ],
        )
        generate_task = gpt4.generate(input_file_id=file.id)
        generate_task.wait()
        output = generate_task.output
        assert len(output.blocks) == 1
        for block in output.blocks:
            usage_found = False
            for tag in block.tags:
                if tag.kind == "token_usage":
                    usage_found = True
            assert usage_found


def test_stopwords():
    with Steamship.temporary_workspace() as steamship:
        gpt4 = steamship.use_plugin(GENERATOR_HANDLE)
        file = File.create(
            steamship,
            blocks=[
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
            ],
        )
        generate_task = gpt4.generate(
            input_file_id=file.id, options={"stop": ["6", "7"]}
        )
        generate_task.wait()
        output = generate_task.output
        assert len(output.blocks) == 1
        for block in output.blocks:
            assert block.text.strip() == "5"
            usage_found = False
            for tag in block.tags:
                if tag.kind == "token_usage":
                    usage_found = True
            assert usage_found
