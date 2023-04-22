from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
from transformers import pipeline




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default="", type=str, help="text prompt")
    parser.add_argument('--workspace', default="trial", type=str, help="workspace")
    parser.add_argument('--model', default='vlt5', type=str, help="model choices - vlt5, bert, XLNet")

    opt = parser.parse_args()

    if opt.model == "vlt5":
        tokenizer = AutoTokenizer.from_pretrained("Voicelab/vlt5-base-keywords")
        model = AutoModelForSeq2SeqLM.from_pretrained("Voicelab/vlt5-base-keywords")

        task_prefix = "Keywords: "
        inputs = [
        opt.text
        ]

        for sample in inputs:
            input_sequences = [task_prefix + sample]
            input_ids = tokenizer(
                input_sequences, return_tensors="pt", truncation=True
            ).input_ids
            output = model.generate(input_ids, no_repeat_ngram_size=3, num_beams=4)
            predicted = tokenizer.decode(output[0], skip_special_tokens=True)
            print(sample, "\n --->", predicted)

    elif opt.mdoel == "bert":
        tokenizer = AutoTokenizer.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")
        model = AutoModelForTokenClassification.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")



    elif opt.model == "XLNet":
        tokenizer = AutoTokenizer.from_pretrained("jasminejwebb/KeywordIdentifier")
        model = AutoModelForTokenClassification.from_pretrained("jasminejwebb/KeywordIdentifier")

        text = opt.text
        inputs = tokenizer(text, return_tensors="pt")

        # Classify tokens
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)

        # Print the predicted labels
        print(predictions.tolist())
