import torch

class IndustryDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, tokenizer, is_train=True):
    self.dataset = dataset
    self.text = self.dataset["text"]
    self.tokenizer = tokenizer
    self.is_train = is_train
    if is_train:
      self.labels = self.dataset["label"]

  def __getitem__(self, idx):
    text = self.text[idx]
    item = self.tokenizer(
        text,
        max_length = 80,
        padding = "max_length",
        truncation=True,
        return_tensors = "pt",
        add_special_tokens=True,
        return_token_type_ids=False
        )
    if self.is_train:
      labels = self.labels[idx]
      item['labels'] = torch.tensor(labels)
    item["input_ids"] = item["input_ids"].squeeze(0)
    item["attention_mask"] = item["attention_mask"].squeeze(0)
    return item

  def __len__(self):
    return len(self.dataset)