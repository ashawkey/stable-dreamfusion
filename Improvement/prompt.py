import textwrap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
from transformers import pipeline
import argparse
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# python prompt_processing.py --text a dog is flying and the rabbit is jumping --model vlt5

if __name__ == '__main__':

    # Mimic the calling part of the main, using
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
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            #print(sample, "\n --->", output_text)

    elif opt.model == "bert":
        tokenizer = AutoTokenizer.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")
        model = AutoModelForTokenClassification.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")

        text = opt.text
        input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")

        # Classify tokens
        outputs = model(input_ids)
        predictions = outputs.logits.detach().numpy()[0]
        labels = predictions.argmax(axis=1)
        labels = labels[1:-1]
        
        print(labels)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = tokens[1:-1]
        output_tokens = [tokens[i] for i in range(len(tokens)) if labels[i] != 0]
        output_text = tokenizer.convert_tokens_to_string(output_tokens)

        #print(output_text)


    elif opt.model == "XLNet":
        tokenizer = AutoTokenizer.from_pretrained("jasminejwebb/KeywordIdentifier")
        model = AutoModelForTokenClassification.from_pretrained("jasminejwebb/KeywordIdentifier")

        text = opt.text
        input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")

        # Classify tokens
        outputs = model(input_ids)
        predictions = outputs.logits.detach().numpy()[0]
        labels = predictions.argmax(axis=1)
        labels = labels[1:-1]
        
        print(labels)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = tokens[1:-1]
        output_tokens = [tokens[i] for i in range(len(tokens)) if labels[i] != 0]
        output_text = tokenizer.convert_tokens_to_string(output_tokens)

        #print(output_text)

wrapped_text = textwrap.fill(output_text, width=50)


print('+' + '-'*52 + '+')
for line in wrapped_text.split('\n'):
    print('| {} |'.format(line.ljust(50)))
print('+' + '-'*52 + '+')
#print(result)
