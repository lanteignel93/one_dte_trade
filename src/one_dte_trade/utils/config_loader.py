import datetime
import json
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Dict, List, Tuple, Type, TypeVar, get_args, get_origin

# Define a generic type variable for dataclasses
T = TypeVar("T")


def _parse_value(target_type: Any, value: Any, field_name: str = "") -> Any:
    """
    Recursively parses a value from JSON-like data to match the target_type
    of a dataclass field. Handles common types like datetime.date, List, and Tuples.
    """
    origin_type = get_origin(target_type)

    if value is None:
        return None

    if origin_type is list or origin_type is List:
        if not isinstance(value, list):
            raise ValueError(
                f"Field '{field_name}': Expected a list for type {target_type}, got {type(value)}"
            )

        list_item_type_args = get_args(target_type)
        list_item_type = list_item_type_args[0] if list_item_type_args else Any
        return [
            _parse_value(list_item_type, item, f"{field_name}[{i}]")
            for i, item in enumerate(value)
        ]

    elif origin_type is tuple or origin_type is Tuple:
        if not isinstance(value, list):  # JSON loads tuples as lists
            raise ValueError(
                f"Field '{field_name}': Expected a list (to be converted to tuple) for type {target_type}, got {type(value)}"
            )

        tuple_item_types = get_args(target_type)
        if (
            tuple_item_types and Ellipsis not in tuple_item_types
        ):  # e.g. Tuple[int, int, int]
            if len(tuple_item_types) != len(value):
                raise ValueError(
                    f"Field '{field_name}': Tuple length mismatch. Expected {len(tuple_item_types)} items for {target_type}, got {len(value)} items."
                )
            return tuple(
                _parse_value(tuple_item_types[i], value[i], f"{field_name}[{i}]")
                for i in range(len(value))
            )
        elif (
            tuple_item_types and tuple_item_types[-1] is Ellipsis
        ):  # e.g. Tuple[int, ...]
            return tuple(
                _parse_value(tuple_item_types[0], item, f"{field_name}[{i}]")
                for i, item in enumerate(value)
            )
        else:  # General Tuple or Tuple without specified inner types
            return tuple(value)

    elif target_type == datetime.date and isinstance(value, str):
        try:
            return datetime.datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(
                f"Field '{field_name}': Could not parse date string '{value}'. Expected YYYY-MM-DD format."
            )

    elif target_type == Any:
        return value

    if not isinstance(value, target_type):
        try:
            return target_type(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Field '{field_name}': Type mismatch or conversion error for value '{value}' (type: {type(value)}) to {target_type}. Error: {e}"
            )

    return value


def load_config_from_json_data(
    dataclass_type: Type[T], config_data: Dict[str, Any]
) -> T:
    """
    Instantiates a dataclass from a dictionary (parsed from JSON),
    performing necessary type conversions.

    Args:
        dataclass_type: The dataclass type to instantiate (e.g., StraddleDataConfig).
        config_data: A dictionary containing the configuration data for the dataclass.

    Returns:
        An instance of the provided dataclass_type.
    """
    if not is_dataclass(dataclass_type):
        raise TypeError(f"{dataclass_type.__name__} is not a dataclass.")

    init_kwargs = {}
    json_keys_processed = set()

    for field_obj in fields(dataclass_type):
        field_name = field_obj.name
        field_type = field_obj.type

        if field_name in config_data:
            json_keys_processed.add(field_name)
            raw_value = config_data[field_name]
            try:
                init_kwargs[field_name] = _parse_value(
                    field_type, raw_value, field_name
                )
            except ValueError as e:
                raise ValueError(
                    f"Error processing field '{field_name}' from JSON data for {dataclass_type.__name__}: {e}"
                )
        elif field_obj.default is MISSING and field_obj.default_factory is MISSING:
            raise ValueError(
                f"Missing required field '{field_name}' for {dataclass_type.__name__} in JSON data and no default defined."
            )

    unexpected_keys = set(config_data.keys()) - json_keys_processed
    if unexpected_keys:
        print(
            f"Warning: Unexpected keys found in JSON data for {dataclass_type.__name__}: {unexpected_keys}. These will be ignored for this dataclass."
        )

    try:
        return dataclass_type(**init_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Error instantiating {dataclass_type.__name__} with processed arguments {init_kwargs}. "
            f"Check for missing required fields or type mismatches. Original error: {e}"
        )


def load_all_project_configs(
    master_json_path: str, config_class_map: Dict[str, Type]
) -> Dict[str, Any]:
    """
    Loads multiple dataclass configurations from a single master JSON file.
    The master JSON file should have top-level keys matching the names in config_class_map.

    Args:
        master_json_path: Path to the main JSON configuration file.
        config_class_map: A dictionary mapping top-level keys in the JSON to their
                          corresponding dataclass types.
                          Example: {"StraddleDataConfig": StraddleDataConfig, ...}

    Returns:
        A dictionary where keys are the same as in config_class_map (can be used as is,
        or you can lowercase them for attribute-like access if preferred) and values are
        the instantiated dataclass objects.
    """
    try:
        with open(master_json_path, "r") as f:
            all_configs_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Master JSON config file not found at {master_json_path}"
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {master_json_path}: {e}")

    loaded_configs = {}
    for config_key, dataclass_type in config_class_map.items():
        if config_key in all_configs_data:
            config_data_dict = all_configs_data[config_key]
            loaded_configs[config_key] = load_config_from_json_data(
                dataclass_type, config_data_dict
            )
        else:
            print(
                f"Warning: Configuration section '{config_key}' not found in {master_json_path}. "
                f"Instantiating {dataclass_type.__name__} with defaults."
            )
            try:
                loaded_configs[config_key] = (
                    dataclass_type()
                )  # Instantiate with defaults
            except TypeError as e:
                print(
                    f"Error: Could not instantiate {dataclass_type.__name__} with defaults due to missing required fields. Error: {e}"
                )
                # Optionally re-raise or handle more gracefully
    return loaded_configs
