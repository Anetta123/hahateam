import torch
from transformers import AutoTokenizer, AutoModel
from langchain_text_splitters import TokenTextSplitter


tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-bge-m3")
embed_model = AutoModel.from_pretrained("deepvk/USER-bge-m3")
embed_model.eval()

large_splitter = TokenTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer, chunk_size=8000, chunk_overlap=128
)
small_splitter = TokenTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer, chunk_size=512, chunk_overlap=64
)


def late_chunking(model_output: torch.Tensor, span_annotations: list):
    token_embeddings = model_output.last_hidden_state  # shape: [batch_size, seq_length, hidden_size]
    outputs = []
    for batch in token_embeddings:
        pooled_embeddings = []
        for start, end in span_annotations:
            if start < end:
                # Среднее по токенам в чанке
                pooled = batch[start:end].mean(dim=0)
                pooled_embeddings.append(pooled.detach().cpu().numpy())
        outputs += pooled_embeddings
    return outputs


def process_large_text(input_text):
    all_chunks = []
    all_embeddings = []

    for text in large_splitter.split_text(input_text):
        text_chunks = small_splitter.split_text(text)
        all_chunks += text_chunks
        span_annotations = []
        current_token = 0
        for chunk in text_chunks:
            encoded = tokenizer(chunk, return_tensors='pt')
            num_tokens = encoded.input_ids.shape[1]
            span_annotations.append((current_token, current_token + num_tokens))
            current_token += num_tokens

        batch_text = " ".join(text_chunks)
        with torch.no_grad():
            inputs = tokenizer(batch_text, return_tensors='pt')
            model_output = embed_model(**inputs)

        all_embeddings += late_chunking(model_output, span_annotations)

    return all_chunks, all_embeddings
