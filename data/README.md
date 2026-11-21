# Data

No data is stored in this repository. The notebook downloads or expects:

| Stage | Data | Source |
|---|---|---|
| 1. Training from random weights | Any plain-text book, one file | The original run used a novel that cannot be redistributed. A public-domain text works the same way, for example *The Verdict* by Edith Wharton, which the book this notebook follows also uses. |
| 2. Pretrained weights | OpenAI GPT-2 774M checkpoint | Downloaded by `gpt_download3.py` into `gpt2/774M/` |
| 3. Spam classification | SMS Spam Collection | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| 4. Instruction fine-tuning | 1,100 instruction and response pairs | [`instruction-data.json`](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json) from the book's repository |
