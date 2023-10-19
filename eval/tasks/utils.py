import re
import nltk
from nltk.sem import logic
from nltk.sem import Expression

logic._counter._value = 0
read_expr = Expression.fromstring
prover = nltk.Prover9(10)


def convert_to_nltk_rep(logic_formula):
    translation_map = {
        "∀": "all ",
        "∃": "exists ",
        "→": "->",
        "¬": "-",
        "∧": "&",
        "∨": "|",
        "⟷": "<->",
        "↔": "<->",
        "0": "Zero",
        "1": "One",
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
        ".": "Dot",
        "Ś": "S",
        "ą": "a",
        "’": "",
    }

    constant_pattern = r'\b([a-z]{2,})(?!\()'
    logic_formula = re.sub(constant_pattern, lambda match: match.group(1).capitalize(), logic_formula)

    for key, value in translation_map.items():
        logic_formula = logic_formula.replace(key, value)

    quant_pattern = r"(all\s|exists\s)([a-z])"
    def replace_quant(match):
        return match.group(1) + match.group(2) + "."
    logic_formula = re.sub(quant_pattern, replace_quant, logic_formula)

    dotted_param_pattern = r"([a-z])\.(?=[a-z])"
    def replace_dotted_param(match):
        return match.group(1)
    logic_formula = re.sub(dotted_param_pattern, replace_dotted_param, logic_formula)

    simple_xor_pattern = r"(\w+\([^()]*\)) ⊕ (\w+\([^()]*\))"
    def replace_simple_xor(match):
        return ("((" + match.group(1) + " & -" + match.group(2) + ") | (-" + match.group(1) + " & " + match.group(2) + "))")
    logic_formula = re.sub(simple_xor_pattern, replace_simple_xor, logic_formula)

    complex_xor_pattern = r"\((.*?)\)\) ⊕ \((.*?)\)\)"
    def replace_complex_xor(match):
        return ("(((" + match.group(1) + ")) & -(" + match.group(2) + "))) | (-(" + match.group(1) + ")) & (" + match.group(2) + "))))")
    logic_formula = re.sub(complex_xor_pattern, replace_complex_xor, logic_formula)

    special_xor_pattern = r"\(\(\((.*?)\)\)\) ⊕ (\w+\([^()]*\))"
    def replace_special_xor(match):
        return ("(((" + match.group(1) + ")) & -" + match.group(2) + ") | (-(" + match.group(1) + ")) & " + match.group(2) + ")")
    logic_formula = re.sub(special_xor_pattern, replace_special_xor, logic_formula)
    
    return logic_formula

def get_all_variables(text):
    pattern = r'\([^()]+\)'
    matches = re.findall(pattern, text)
    all_variables = []
    for m in matches:
        m = m[1:-1]
        m = m.split(",")
        all_variables += [i.strip() for i in m]
    return list(set(all_variables))

def reformat_fol(fol):
    translation_map = {
        "0": "Zero", 
        "1": "One",
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
        ".": "Dot",
        "’": "",
        "-": "_",
        "'": "",
        " ": "_"
    }
    all_variables = get_all_variables(fol)
    for variable in all_variables:
        variable_new = variable[:]
        for k, v in translation_map.items():
            variable_new = variable_new.replace(k, v)
        fol = fol.replace(variable, variable_new)
    return fol

def evaluate(premises, conclusion):
    premises = [reformat_fol(p) for p in premises]
    conclusion = reformat_fol(conclusion)

    c = read_expr(conclusion)
    p_list = []
    for p in premises:
        p_list.append(read_expr(p))
    truth_value = prover.prove(c, p_list)
    if truth_value:
        return "True"
    else:
        neg_c = read_expr("-(" + conclusion + ")")
        negation_true = prover.prove(neg_c, p_list)
        if negation_true:
            return "False"
        else:
            return "Uncertain"
