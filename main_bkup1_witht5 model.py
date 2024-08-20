import os
import numpy as np
import torch
import re
import json
import argparse
import random
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration, T5Config
from model import T5ForMultimodalGeneration
from utils_data import img_shape, load_data_std, load_data_img, ScienceQADatasetStd, ScienceQADatasetImg
from utils_prompt import *
from utils_evaluate import get_scores
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
import evaluate
import inspect
from optimum.pipelines import pipeline
#from optimum.pruning import prune_model

import torch
import torch.nn.utils.prune as prune


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=1)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet','vit'], help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args

# def T5Trainer(
#     dataframe, args,
# ):
#     print("T5 trainer>>>>>>")
#     torch.manual_seed(args.seed)  # pytorch random seed
#     np.random.seed(args.seed)  # numpy random seed
#     torch.backends.cudnn.deterministic = True
    
#     print("line>>",inspect.currentframe().f_lineno)
#     if args.evaluate_dir is not None:
#         print("Inside if of args>>>>>",inspect.currentframe().f_lineno)
#         args.model = args.evaluate_dir
#     print("line>>",inspect.currentframe().f_lineno)
#     tokenizer = AutoTokenizer.from_pretrained(args.model)
#     print("line>>",inspect.currentframe().f_lineno)
#     console.log(f"""[Model]: Loading {args.model}...\n""")
#     print("line>>",inspect.currentframe().f_lineno)
#     console.log(f"[Data]: Reading data...\n")
#     print("line>>",inspect.currentframe().f_lineno)
#     problems = dataframe['problems']
#     print("line>>",inspect.currentframe().f_lineno)
#     qids = dataframe['qids']
#     print("line>>",inspect.currentframe().f_lineno)
#     train_qids = qids['train']
#     print("line>>",inspect.currentframe().f_lineno)
#     test_qids = qids['test']
#     print("line>>",inspect.currentframe().f_lineno)
#     val_qids = qids['val']
    
#     if args.evaluate_dir is not None:
#         save_dir = args.evaluate_dir
#     else:
#         model_name = args.model.replace("/","-")
#         gpu_count = torch.cuda.device_count()
#         save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
#         if not os.path.exists(save_dir):
#             os.mkdir(save_dir)
#     print(save_dir)

#     # Load and configure the model
#     if args.img_type is not None:
#         patch_size = img_shape[args.img_type]
#         config = T5Config.from_pretrained('t5-small')  # Switch to smaller model
#         config.num_layers = 6  # Reduce number of layers
#         model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size, config=config)
#         name_maps = dataframe['name_maps']
#         image_features = dataframe['image_features']
#         train_set = ScienceQADatasetImg(
#             problems,
#             train_qids,
#             name_maps,
#             tokenizer,
#             args.input_len,
#             args.output_len,
#             args,
#             image_features,
#         )
#         eval_set = ScienceQADatasetImg(
#             problems,
#             val_qids,
#             name_maps,
#             tokenizer,
#             args.input_len,
#             args.output_len,
#             args,
#             image_features,
#             args.eval_le,
#         )
#         test_set = ScienceQADatasetImg(
#             problems,
#             test_qids,
#             name_maps,
#             tokenizer,
#             args.input_len,
#             args.output_len,
#             args,
#             image_features,
#             args.test_le,
#         )
#     else:
#         config = T5Config.from_pretrained('t5-small')  # Switch to smaller model
#         config.num_layers = 6  # Reduce number of layers
#         model = T5ForConditionalGeneration(config)
#         train_set = ScienceQADatasetStd(
#             problems,
#             train_qids,
#             tokenizer,
#             args.input_len,
#             args.output_len,
#             args,
#         )
#         eval_set = ScienceQADatasetStd(
#             problems,
#             val_qids,
#             tokenizer,
#             args.input_len,
#             args.output_len,
#             args,
#             args.eval_le,
#         )
#         test_set = ScienceQADatasetStd(
#             problems,
#             test_qids,
#             tokenizer,
#             args.input_len,
#             args.output_len,
#             args,
#             args.test_le,
#         )

#     # # Apply model pruning
#     # if args.img_type is None:
#     #     model = prune_model(model, sparsity=0.5)  # Prune 50% of the model weights

#     # Apply model pruning
#     if args.img_type is None:
#         # Iterate over model's parameters and apply pruning
#         for name, module in model.named_modules():
#             if isinstance(module, torch.nn.Linear):
#                 prune.random_unstructured(module, name="weight", amount=0.5)  # Prune 50% of the weights


#     # Configure data collator
#     datacollator = DataCollatorForSeq2Seq(tokenizer)
#     print("model parameters: ", model.num_parameters())

#     def extract_ans(ans):
#         pattern = re.compile(r'The answer is \(([A-Z])\)')
#         res = pattern.findall(ans)
#         if len(res) == 1:
#             answer = res[0]  # 'A', 'B', ...
#         else:
#             answer = "FAILED"
#         return answer  

#     def compute_metrics_acc(eval_preds):
#         if args.use_generate:
#             preds, targets = eval_preds
#             if isinstance(preds, tuple):
#                 preds = preds[0]
#         else:
#             preds = eval_preds.predictions[0]
#             targets = eval_preds.label_ids
#             preds = preds.argmax(axis=2)
#         preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#         targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#         correct = 0
#         assert len(preds) == len(targets)
#         for idx, pred in enumerate(preds):
#             reference = targets[idx]
#             reference = extract_ans(reference)
#             extract_pred = extract_ans(pred)
#             best_option = extract_pred
#             if reference == best_option:
#                 correct +=1 
#         return {'accuracy': 1.0*correct/len(targets)}
    
#     metric = evaluate.load("rouge")
#     def postprocess_text(preds, labels):
#         preds = [pred.strip() for pred in preds]
#         labels = [label.strip() for label in labels]
#         preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
#         labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
#         return preds, labels

#     def compute_metrics_rougel(eval_preds):
#         if args.use_generate:
#             preds, targets = eval_preds
#             if isinstance(preds, tuple):
#                 preds = preds[0]
#         else:
#             preds = eval_preds.predictions[0]
#             targets = eval_preds.label_ids
#             preds = preds.argmax(axis=2)
#         preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#         targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#         preds, targets = postprocess_text(preds, targets)
#         results = metric.compute(predictions=preds, references=targets)
#         return results

#     def compute_metrics(eval_preds):
#         metrics_acc = compute_metrics_acc(eval_preds)
#         metrics_rougel = compute_metrics_rougel(eval_preds)
#         return {**metrics_acc, **metrics_rougel}

#     # Load the pre-trained model
#     if args.evaluate_dir is not None:
#         print("Loading existing model...")
#         model = T5ForConditionalGeneration.from_pretrained(args.evaluate_dir)
#     else:
#         print("Training model...")
#         training_args = Seq2SeqTrainingArguments(
#             output_dir=save_dir,
#             evaluation_strategy="steps",
#             save_total_limit=1,
#             save_strategy="steps",
#             logging_steps=500,
#             per_device_train_batch_size=args.bs,
#             per_device_eval_batch_size=args.eval_bs,
#             gradient_accumulation_steps=args.eval_acc,
#             learning_rate=args.lr,
#             num_train_epochs=args.epoch,
#             predict_with_generate=not args.use_generate,
#             remove_unused_columns=False,
#             report_to='tensorboard'
#         )
#         print(f"Actual number of epochs>>>>>: {training_args.num_train_epochs}")

#         trainer = Seq2SeqTrainer(
#             model=model,
#             args=training_args,
#             data_collator=datacollator,
#             train_dataset=train_set,
#             eval_dataset=eval_set,
#             compute_metrics=compute_metrics
#         )

#         if not args.final_eval:
#             trainer.train()

#         trainer.save_model(save_dir)
#         tokenizer.save_pretrained(save_dir)
#         print(f"Model saved to {save_dir}")
    
#     # Evaluate model
#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         data_collator=datacollator,
#         eval_dataset=eval_set,
#         compute_metrics=compute_metrics
#     )

#     eval_results = trainer.evaluate()
#     print(f"Eval results: {eval_results}")

#     # Test model
#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         data_collator=datacollator,
#         eval_dataset=test_set,
#         compute_metrics=compute_metrics
#     )

#     test_results = trainer.evaluate()
#     print(f"Test results: {test_results}")

def T5Trainer(dataframe, args):
    print("T5 trainer>>>>>>")
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    print("line>>",inspect.currentframe().f_lineno)
    if args.evaluate_dir is not None:
        print("Inside if of args>>>>>",inspect.currentframe().f_lineno)
        args.model = args.evaluate_dir
    print("line>>",inspect.currentframe().f_lineno)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("line>>",inspect.currentframe().f_lineno)
    console.log(f"""[Model]: Loading {args.model}...\n""")
    print("line>>",inspect.currentframe().f_lineno)
    console.log(f"[Data]: Reading data...\n")
    print("line>>",inspect.currentframe().f_lineno)
    problems = dataframe['problems']
    print("line>>",inspect.currentframe().f_lineno)
    qids = dataframe['qids']
    print("line>>",inspect.currentframe().f_lineno)
    train_qids = qids['train']
    print("line>>",inspect.currentframe().f_lineno)
    test_qids = qids['test']
    print("line>>",inspect.currentframe().f_lineno)
    val_qids = qids['val']
    
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/","-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    print(save_dir)

    # Load and configure the model
    if args.img_type is not None:
        patch_size = img_shape[args.img_type]
        config = T5Config.from_pretrained('t5-small')  # Switch to smaller model
        config.num_layers = 6  # Reduce number of layers
        model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size, config=config)
        name_maps = dataframe['name_maps']
        image_features = dataframe['image_features']
        train_set = ScienceQADatasetImg(
            problems,
            train_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
        )
        eval_set = ScienceQADatasetImg(
            problems,
            val_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.eval_le,
        )
        test_set = ScienceQADatasetImg(
            problems,
            test_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.test_le,
        )
    else:
        config = T5Config.from_pretrained('t5-small')  # Switch to smaller model
        config.num_layers = 6  # Reduce number of layers
        model = T5ForConditionalGeneration(config)
        train_set = ScienceQADatasetStd(
            problems,
            train_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
        )
        eval_set = ScienceQADatasetStd(
            problems,
            val_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.eval_le,
        )
        test_set = ScienceQADatasetStd(
            problems,
            test_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.test_le,
        )

    # Apply model pruning
    if args.img_type is None:
        # Iterate over model's parameters and apply pruning
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.random_unstructured(module, name="weight", amount=0.5)  # Prune 50% of the weights

    # Configure data collator
    datacollator = DataCollatorForSeq2Seq(tokenizer)
    print("model parameters: ", model.num_parameters())

    def extract_ans(ans):
        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(ans)
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED"
        return answer  

    # def compute_metrics_acc(eval_preds):
    #     if args.use_generate:
    #         preds, targets = eval_preds
    #         if isinstance(preds, tuple):
    #             preds = preds[0]
    #     else:
    #         preds = eval_preds.predictions[0]
    #         targets = eval_preds.label_ids
    #         preds = preds.argmax(axis=2)
    #     preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     correct = 0
    #     assert len(preds) == len(targets)
    #     for idx, pred in enumerate(preds):
    #         reference = targets[idx]
    #         reference = extract_ans(reference)
    #         extract_pred = extract_ans(pred)
    #         best_option = extract_pred
    #         if reference == best_option:
    #             correct +=1 
    #     return {'accuracy': 1.0*correct/len(targets)}
    
    # metric = evaluate.load("rouge")
    # def postprocess_text(preds, labels):
    #     preds = [pred.strip() for pred in preds]
    #     labels = [label.strip() for label in labels]
    #     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    #     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    #     return preds, labels

    # def compute_metrics_rougel(eval_preds):
    #     if args.use_generate:
    #         preds, targets = eval_preds
    #         if isinstance(preds, tuple):
    #             preds = preds[0]
    #     else:
    #         preds = eval_preds.predictions[0]
    #         targets = eval_preds.label_ids
    #         preds = preds.argmax(axis=2)
    #     preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     preds, targets = postprocess_text(preds, targets)
    #     results = metric.compute(predictions=preds, references=targets)
    #     return results

    # def compute_metrics(eval_preds):
    #     metrics_acc = compute_metrics_acc(eval_preds)
    #     metrics_rougel = compute_metrics_rougel(eval_preds)
    #     return {**metrics_acc, **metrics_rougel}

    # # Load the pre-trained model
    # if args.evaluate_dir is not None:
    #     print("Loading existing model...")
    #     model = T5ForConditionalGeneration.from_pretrained(args.evaluate_dir)
    # else:
    #     print("Training model...")
    #     training_args = Seq2SeqTrainingArguments(
    #         output_dir=save_dir,
    #         evaluation_strategy="steps",
    #         save_total_limit=1,
    #         save_strategy="steps",
    #         logging_steps=500,
    #         per_device_train_batch_size=args.bs,
    #         per_device_eval_batch_size=args.eval_bs,
    #         gradient_accumulation_steps=args.eval_acc if args.eval_acc is not None else 1,  # Ensure it's an integer
    #         learning_rate=args.lr,
    #         num_train_epochs=args.epoch,
    #         predict_with_generate=not args.use_generate,
    #         remove_unused_columns=False,
    #         report_to='tensorboard'
    #     )
    #     print(f"Actual number of epochs>>>>>: {training_args.num_train_epochs}")

    #     trainer = Seq2SeqTrainer(
    #         model=model,
    #         args=training_args,
    #         data_collator=datacollator,
    #         train_dataset=train_set,
    #         eval_dataset=eval_set,
    #         compute_metrics=compute_metrics
    #     )

    #     if not args.final_eval:
    #         trainer.train()

    #     trainer.save_model(save_dir)
    #     tokenizer.save_pretrained(save_dir)
    #     print(f"Model saved to {save_dir}")
    
    # # Evaluate model
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=datacollator,
    #     eval_dataset=eval_set,
    #     compute_metrics=compute_metrics
    # )

    # eval_results = trainer.evaluate()
    # print(f"Eval results: {eval_results}")

    # # Test model
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=datacollator,
    #     eval_dataset=test_set,
    #     compute_metrics=compute_metrics
    # )

    # test_results = trainer.evaluate()
    # print(f"Test results: {test_results}")

    # def extract_ans(ans):
    #     pattern = re.compile(r'The answer is \(([A-Z])\)')
    #     res = pattern.findall(ans)
    #     if len(res) == 1:
    #         return res[0]  # 'A', 'B', ...
    #     return "FAILED"

    # def compute_metrics_acc(eval_preds):
    #     if args.use_generate:
    #         preds, targets = eval_preds
    #         if isinstance(preds, tuple):
    #             preds = preds[0]
    #     else:
    #         preds = eval_preds.predictions[0]
    #         targets = eval_preds.label_ids
    #         print(f"Preds shape: {preds.shape}")
    #         print(f"Targets shape: {targets.shape}")
            
    #         # Adjust axis based on the actual shape
    #         if preds.ndim > 1 and preds.shape[1] > 1:
    #             preds = np.argmax(preds, axis=-1)  # Use -1 for the last dimension

    #     preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
    #     correct = 0
    #     assert len(preds) == len(targets)
    #     for idx, pred in enumerate(preds):
    #         reference = targets[idx]
    #         reference = extract_ans(reference)
    #         extract_pred = extract_ans(pred)
    #         best_option = extract_pred
    #         if reference == best_option:
    #             correct += 1 
    #     return {'accuracy': 1.0 * correct / len(targets)}

    # metric = evaluate.load("rouge")

    # def postprocess_text(preds, labels):
    #     preds = [pred.strip() for pred in preds]
    #     labels = [label.strip() for label in labels]
    #     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    #     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    #     return preds, labels

    # def compute_metrics_rougel(eval_preds):
    #     if args.use_generate:
    #         preds, targets = eval_preds
    #         if isinstance(preds, tuple):
    #             preds = preds[0]
    #     else:
    #         preds = eval_preds.predictions[0]
    #         targets = eval_preds.label_ids
    #         if preds.ndim > 1 and preds.shape[1] > 1:
    #             preds = np.argmax(preds, axis=-1)  # Use -1 for the last dimension
        
    #     preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     preds, targets = postprocess_text(preds, targets)
    #     results = metric.compute(predictions=preds, references=targets)
    #     return results

    # def compute_metrics(eval_preds):
    #     metrics_acc = compute_metrics_acc(eval_preds)
    #     metrics_rougel = compute_metrics_rougel(eval_preds)
    #     return {**metrics_acc, **metrics_rougel}

    # # Load the pre-trained model
    # if args.evaluate_dir is not None:
    #     print("Loading existing model...")
    #     model = T5ForConditionalGeneration.from_pretrained(args.evaluate_dir)
    # else:
    #     print("Training model...")
    #     training_args = Seq2SeqTrainingArguments(
    #         output_dir=save_dir,
    #         evaluation_strategy="steps",
    #         save_total_limit=1,
    #         save_strategy="steps",
    #         logging_steps=500,
    #         per_device_train_batch_size=args.bs,
    #         per_device_eval_batch_size=args.eval_bs,
    #         gradient_accumulation_steps=args.eval_acc if args.eval_acc is not None else 1,  # Ensure it's an integer
    #         learning_rate=args.lr,
    #         num_train_epochs=args.epoch,
    #         predict_with_generate=not args.use_generate,
    #         remove_unused_columns=False,
    #         report_to='tensorboard'
    #     )
    #     print(f"Actual number of epochs>>>>>: {training_args.num_train_epochs}")

    #     trainer = Seq2SeqTrainer(
    #         model=model,
    #         args=training_args,
    #         data_collator=datacollator,
    #         train_dataset=train_set,
    #         eval_dataset=eval_set,
    #         compute_metrics=compute_metrics
    #     )

    #     if not args.final_eval:
    #         trainer.train()

    #     trainer.save_model(save_dir)
    #     tokenizer.save_pretrained(save_dir)
    #     print(f"Model saved to {save_dir}")

    # # Evaluate model
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=datacollator,
    #     eval_dataset=eval_set,
    #     compute_metrics=compute_metrics
    # )

    # eval_results = trainer.evaluate()
    # print(f"Eval results: {eval_results}")

    # # Test model
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=datacollator,
    #     eval_dataset=test_set,
    #     compute_metrics=compute_metrics
    # )

    # test_results = trainer.evaluate()
    # print(f"Test results: {test_results}")

    def compute_metrics_acc(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            print(f"Preds shape: {preds.shape}")
            print(f"Targets shape: {targets.shape}")
            
    # Check if preds has more than one dimension before using argmax on axis 2
    if len(preds.shape) > 1:
        preds = preds.argmax(axis=-1)  # Apply argmax only if more than one dimension

        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct += 1 
        return {'accuracy': 1.0 * correct / len(targets)}

    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            
        preds = np.argmax(preds, axis=-1)  # Adjust axis based on shape
        
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        preds, targets = postprocess_text(preds, targets)
        results = metric.compute(predictions=preds, references=targets)
        return results

    def compute_metrics(eval_preds):
        metrics_acc = compute_metrics_acc(eval_preds)
        metrics_rougel = compute_metrics_rougel(eval_preds)
        return {**metrics_acc, **metrics_rougel}

    # Load the pre-trained model
    if args.evaluate_dir is not None:
        print("Loading existing model...")
        model = T5ForConditionalGeneration.from_pretrained(args.evaluate_dir)
    else:
        print("Training model...")
        training_args = Seq2SeqTrainingArguments(
            output_dir=save_dir,
            evaluation_strategy="steps",
            save_total_limit=1,
            save_strategy="steps",
            logging_steps=500,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            gradient_accumulation_steps=args.eval_acc if args.eval_acc is not None else 1,  # Ensure it's an integer
            learning_rate=args.lr,
            num_train_epochs=args.epoch,
            predict_with_generate=not args.use_generate,
            remove_unused_columns=False,
            report_to='tensorboard'
        )
        print(f"Actual number of epochs>>>>>: {training_args.num_train_epochs}")

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=datacollator,
            train_dataset=train_set,
            eval_dataset=eval_set,
            compute_metrics=compute_metrics
        )

        if not args.final_eval:
            trainer.train()

        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")

    # Evaluate model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=datacollator,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics
    )

    eval_results = trainer.evaluate()
    print(f"Eval results: {eval_results}")

    # Test model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=datacollator,
        eval_dataset=test_set,
        compute_metrics=compute_metrics
    )

    test_results = trainer.evaluate()
    print(f"Test results: {test_results}")

if __name__ == '__main__':
    args = parse_args()
    print("line>>",inspect.currentframe().f_lineno)
    print("inside else of main  statement>>>>.")
    problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
    dataframe = {'problems':problems, 'qids':qids}

    T5Trainer(dataframe, args)
