from copy import copy, deepcopy
import inspect
from typing import Any, Dict, List, Optional, Text, Tuple, Type

from rasa.architecture_prototype.graph_components import (
    DomainReader,
    MessageToE2EFeatureConverter,
    StoryGraphReader,
    StoryToTrainingDataConverter,
    TrackerGenerator,
    TrainingDataReader,
)
from rasa.architecture_prototype import graph
from rasa.nlu import registry
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.components import Component
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.utils.io import read_yaml
import rasa.core.registry


def train_and_process_component(
    component_class: Type[Component],
    input_task_name: Text,
    config: Optional[Dict[Text, Any]] = None,
    name: Optional[Text] = None,
    train_function: Text = "train",
    process_function: Text = "process_training_data",
    input_data_param_name: Text = "training_data",
) -> Tuple[Dict[Text, Any], Text]:
    name = name if name else component_class.name
    train_task_name = f"train_{name}"
    process_task_name = f"process_{name}"
    config = config if config else {}
    return (
        {
            train_task_name: {
                "uses": component_class,
                "fn": train_function,
                "config": config,
                "needs": {input_data_param_name: input_task_name},
            },
            process_task_name: {
                "uses": component_class,
                "fn": process_function,
                "config": config,
                "needs": {
                    "resource_name": train_task_name,
                    input_data_param_name: input_task_name,
                },
            },
        },
        process_task_name,
    )


def train_component(
    component_class: Type[Component],
    input_task_name: Text,
    config: Optional[Dict[Text, Any]] = None,
    name: Optional[Text] = None,
    train_function: Text = "train",
    input_data_param_name: Text = "training_data",
) -> Tuple[Dict[Text, Any], Text]:
    name = name if name else component_class.name
    train_task_name = f"train_{name}"
    config = config if config else {}
    return (
        {
            train_task_name: {
                "uses": component_class,
                "fn": train_function,
                "config": config,
                "needs": {input_data_param_name: input_task_name},
            },
        },
        input_task_name,
    )


def process_component(
    component_class: Type[Component],
    input_task_name: Text,
    config: Optional[Dict[Text, Any]] = None,
    name: Optional[Text] = None,
    process_function: Text = "process_training_data",
    input_data_param_name: Text = "training_data",
) -> Tuple[Dict[Text, Any], Text]:
    name = name if name else component_class.name
    process_task_name = f"process_{name}"
    config = config if config else {}
    return (
        {
            process_task_name: {
                "uses": component_class,
                "fn": process_function,
                "config": config,
                "needs": {input_data_param_name: input_task_name,},
                "persist": False,
            },
        },
        process_task_name,
    )


def nlu_config_to_train_graph_schema(
    project: Text,
    config: Dict[Text, Any],
    component_namespace: Optional[Text] = None,
    input_task: Optional[Text] = None,
    only_process: bool = False,
) -> Tuple[Dict[Text, Any], Text]:
    nlu_pipeline = deepcopy(config["pipeline"])
    # TODO: get this information from the class?
    meta: Dict[Type[Component], Text] = {
        WhitespaceTokenizer: "process",
        RegexFeaturizer: "train_process",
        LexicalSyntacticFeaturizer: "train_process",
        CountVectorsFeaturizer: "train_process",
        DIETClassifier: "train",
        ResponseSelector: "train",
        EntitySynonymMapper: "train",
    }
    if input_task:
        last_component_out = input_task
        nlu_train_graph = {}
    else:
        last_component_out = "load_data"
        nlu_train_graph = {
            "load_data": {
                "uses": TrainingDataReader,
                "fn": "read",
                "config": {"project": project},
                "needs": {},
                "persist": False,
            },
        }
    for i, component in enumerate(nlu_pipeline):
        component_name = component.pop("name")
        unique_component_name = f"{component_name}_{i}"
        if component_namespace:
            unique_component_name = f"{component_namespace}_{unique_component_name}"
        component_class = registry.get_component_class(component_name)
        step_type = meta[component_class]
        if step_type == "train" and only_process:
            continue
        config = {"component_config": component}
        builder = {
            "process": process_component,
            "train": train_component,
            "train_process": train_and_process_component,
        }[step_type]
        component_def, last_component_out = builder(
            component_class=component_class,
            input_task_name=last_component_out,
            name=unique_component_name,
            config=config,
        )
        nlu_train_graph.update(component_def)

    graph.fill_defaults(nlu_train_graph)

    return nlu_train_graph, last_component_out


def core_config_to_train_graph_schema(
    project: Text, config: Dict[Text, Any]
) -> Tuple[Dict[Text, Any], List[Text]]:
    policies = deepcopy(config["policies"])
    core_train_graph = {
        "load_domain": {
            "uses": DomainReader,
            "fn": "read",
            "config": {"project": project},
            "needs": {},
        },
        "load_stories": {
            "uses": StoryGraphReader,
            "fn": "read",
            "config": {"project": project},
            "needs": {},
            "persist": False,
        },
        "generate_trackers": {
            "uses": TrackerGenerator,
            "fn": "generate",
            "config": {},
            "needs": {"domain": "load_domain", "story_graph": "load_stories"},
            "persist": False,
        },
    }
    policy_names = []
    e2e = False
    for i, policy in enumerate(policies):
        policy_name = policy.pop("name")
        unique_policy_name = f"{policy_name}_{i}"
        policy_names.append(unique_policy_name)
        policy_class = rasa.core.registry.policy_from_module_path(policy_name)
        policy_step = {
            unique_policy_name: {
                "uses": policy_class,
                "fn": "train",
                "config": policy,
                "needs": {
                    "training_trackers": "generate_trackers",
                    "domain": "load_domain",
                },
            },
        }
        if "e2e_features" in inspect.signature(policy_class.train).parameters:
            policy_step[unique_policy_name]["needs"][
                "e2e_features"
            ] = "create_e2e_lookup"
            e2e = True
        core_train_graph.update(policy_step)

    if e2e:
        core_train_graph["convert_stories_for_nlu"] = {
            "uses": StoryToTrainingDataConverter,
            "fn": "convert_for_training",
            "config": {},
            "needs": {"story_graph": "load_stories"},
            "persist": False,
        }
        nlu_train_graph_schema, nlu_out = nlu_config_to_train_graph_schema(
            project,
            config,
            component_namespace="core",
            input_task="convert_stories_for_nlu",
            only_process=True,
        )
        core_train_graph.update(nlu_train_graph_schema)
        core_train_graph["create_e2e_lookup"] = {
            "uses": MessageToE2EFeatureConverter,
            "fn": "convert",
            "config": {},
            "needs": {"training_data": nlu_out},
            "persist": False,
        }

    graph.fill_defaults(core_train_graph)

    return core_train_graph, policy_names


def old_config_to_graph_schema(
    project: Text, config: Text
) -> Tuple[Dict[Text, Any], List[Text]]:
    config_dict = read_yaml(config)
    nlu_train_graph_schema, nlu_out = nlu_config_to_train_graph_schema(
        project, config_dict
    )
    core_train_graph_schema, core_outs = core_config_to_train_graph_schema(
        project, config_dict
    )
    return {**core_train_graph_schema, **nlu_train_graph_schema}, [*core_outs, nlu_out]
