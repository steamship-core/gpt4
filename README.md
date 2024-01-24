# LiteLLM Plugin for Steamship for OpenAI compatibility

This plugin provides access to LiteLLM's language model translation to OpenAI format for text generation.

## Usage

Use of this plugin is subject to individual models' terms of use, including but not limited to:
- OpenAI's [Terms of Use](https://openai.com/policies/terms-of-use).

## Examples

#### Basic

```python
litellm = steamship.use_plugin("litellm")
task = litellm.generate(text=prompt)
task.wait()
for block in task.output.blocks:
    print(block.text)
```

#### With Runtime Parameters

```python
litellm = steamship.use_plugin("litellm")
task = litellm.generate(text=prompt, options={"stop": ["6", "7"]})
task.wait()
for block in task.output.blocks:
    print(block.text)
```

## Cost

[Pricing page](https://www.steamship.com/plugins/litellm?tab=Pricing)


## Developing this Plugin

Clone this repository, then set up a Python virtual environment with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements.dev.txt
```
