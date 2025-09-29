from pgmpy.estimators import HillClimbSearch, BIC
from pgmpy.estimators import PC

def exdesc_iamb(data, sensitive_feature):
    c_est = PC(data)
    dag_model = c_est.estimate(estimator_type="IAMB", significance_level=0.05)
    desc_of_A = descendants_of(dag_model, sensitive_feature)
    all_features = list(data.columns)
    selected_features = [f for f in all_features if f not in desc_of_A]
    if "target" in selected_features:
        selected_features.remove("target")
    return selected_features


def mb_exdesc_iamb(data, sensitive_feature):
    c_est = PC(data)
    dag_model = c_est.estimate(estimator_type="IAMB", significance_level=0.05)
    dag_model = dag_model.to_dag()
    mb_target = dag_model.get_markov_blanket("target")
    desc_of_A = descendants_of(dag_model, sensitive_feature)
    selected_features = [v for v in mb_target if v not in desc_of_A]
    if "target" in selected_features:
        selected_features.remove("target")
    return selected_features


def descendants_of(model, node):
    desc = set()
    stack = [node]
    while stack:
        current = stack.pop()
        children = model.successors(current)   # immediate children
        new = set(children) - desc
        desc |= new
        stack.extend(new)
    return desc