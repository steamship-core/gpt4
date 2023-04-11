from steamship import Steamship

steamship = Steamship()
gpt4 = steamship.use_plugin("gpt-4", config={"max_tokens":1024})
while True:
	prompt = input("Prompt: ")
	task = gpt4.generate(text=prompt)
	task.wait()
	print(task.output.blocks[0].text)