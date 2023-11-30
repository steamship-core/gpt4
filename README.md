# OpenAI GPT-4 Plugin for Steamship

This plugin provides access to OpenAI's GPT-4 language model for text generation.

## Usage

Use of this plugin is subject to OpenAI's [Terms of Use](https://openai.com/policies/terms-of-use).

## Examples

#### Basic

```python
gpt4 = steamship.use_plugin("gpt-4")
task = gpt4.generate(text=prompt)
task.wait()
for block in task.output.blocks:
    print(block.text)
```

#### With Runtime Parameters

```python
gpt4 = steamship.use_plugin("gpt-4")
task = gpt4.generate(text=prompt, options={"stop": ["6", "7"]})
task.wait()
for block in task.output.blocks:
    print(block.text)
```

## Cost

[Pricing page](https://www.steamship.com/plugins/gpt-4?tab=Pricing)


## Developing this Plugin

Clone this repository, then set up a Python virtual environment with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements.dev.txt
```
