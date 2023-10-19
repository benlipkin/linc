import random
import json
random.seed(548)  # to reproduce the dataset used in the paper

DIR = 'proofwriter-dataset-V2020.12.3/OWA/depth-5'
TARGET_LABELS_PER_TASK_AND_DEPTH = 20

for split in ['train', 'test']:
    current_tasks_allocated = {i: 0 for i in range(0, 6)}
    current_labels_allocated = {'True': 0, 'False': 0, 'Uncertain': 0}
    current_labels_allocated_per_task = {i: {'True': 0, 'False': 0, 'Uncertain': 0} for i in range(0, 6)}
    with open(f'{DIR}/meta-{split}.jsonl', 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            doc = json.loads(line)
            # loop over questions in random order
            questions = list(doc['questions'].keys())
            for q in questions:
                doc['questions'][q]['answer'] = str(doc['questions'][q]['answer'])
            random.shuffle(questions)
            for q in questions:
                question = doc['questions'][q]
                d = question['QDep']
                if d >= 6: continue
                if current_tasks_allocated[d] == len(current_labels_allocated) * TARGET_LABELS_PER_TASK_AND_DEPTH:
                    continue
                if question['answer'] == 'Unknown': question['answer'] = 'Uncertain'  # just to agree with FOLIO
                if current_labels_allocated_per_task[d][question['answer']] == TARGET_LABELS_PER_TASK_AND_DEPTH: continue
                dict_to_write = {'id': doc['id'], 'theory': doc['theory'], 'question': question['question'], 'answer': str(question['answer']), 'QDep': d}
                with open(f'proofwriter-{split}-processed-balanced.jsonl', 'a') as f:
                    f.write(json.dumps(dict_to_write) + '\n')
                current_tasks_allocated[d] += 1
                current_labels_allocated_per_task[d][question['answer']] += 1
                break

with open('proofwriter-test-processed-balanced.jsonl', 'r') as f:
    lines = f.readlines()
    print(f'Number of results: {len(lines)}')
    jsons = [json.loads(line) for line in lines]
    print(f'Number of tasks: {len(set([json["id"] for json in jsons]))}')
    print(f'Number of labels: {len(set([json["answer"] for json in jsons]))}')
    print(f'Counts of each label: {dict([(label, len([json for json in jsons if json["answer"] == label])) for label in set([json["answer"] for json in jsons])])}')
    print(f'Counts of each depth: {dict([(depth, len([json for json in jsons if json["QDep"] == depth])) for depth in set([json["QDep"] for json in jsons])])}')
    print(f'Count of each label at each depth: {dict([(depth, dict([(label, len([json for json in jsons if json["QDep"] == depth and json["answer"] == label])) for label in set([json["answer"] for json in jsons])])) for depth in set([json["QDep"] for json in jsons])])}')
