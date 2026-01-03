"""
Advanced Python Metaclasses Module.

Demonstrates:
- Metaclass fundamentals
- Singleton pattern via metaclass
- Registry pattern for automatic class registration
- Validation metaclass for enforcing constraints
- Abstract base class patterns
"""

from __future__ import annotations

from abc import ABCMeta
from typing import Any, Callable, ClassVar, TypeVar

from loguru import logger

T = TypeVar("T")


class SingletonMeta(type):
    """
    Metaclass implementing the Singleton pattern.

    Ensures only one instance of a class exists across the application.
    Thread-safe implementation using class-level instance storage.

    This is useful for:
    - Configuration managers
    - Database connection pools
    - Spark session managers

    Example:
        >>> class Database(metaclass=SingletonMeta):
        ...     def __init__(self, host: str):
        ...         self.host = host
        >>> db1 = Database("localhost")
        >>> db2 = Database("remote")  # Returns same instance
        >>> db1 is db2  # True
    """

    _instances: ClassVar[dict[type, Any]] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Override instance creation to return singleton.

        If an instance already exists, return it. Otherwise,
        create a new instance and cache it.
        """
        if cls not in cls._instances:
            logger.debug(f"Creating singleton instance of {cls.__name__}")
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            logger.debug(f"Returning existing instance of {cls.__name__}")

        return cls._instances[cls]

    @classmethod
    def clear_instance(mcs, cls: type) -> None:
        """
        Clear the singleton instance (useful for testing).

        Args:
            cls: The class whose instance should be cleared.
        """
        if cls in mcs._instances:
            del mcs._instances[cls]
            logger.debug(f"Cleared singleton instance of {cls.__name__}")


class RegistryMeta(type):
    """
    Metaclass for automatic class registration.

    Automatically registers all subclasses of a base class,
    enabling plugin-like architecture and factory patterns.

    This is useful for:
    - Transformer/Extractor registration
    - Plugin systems
    - Strategy pattern implementations

    Example:
        >>> class BaseTransformer(metaclass=RegistryMeta):
        ...     pass
        >>> class CleansingTransformer(BaseTransformer):
        ...     pass
        >>> class ValidationTransformer(BaseTransformer):
        ...     pass
        >>> BaseTransformer.get_registry()
        {'CleansingTransformer': <class 'CleansingTransformer'>, ...}
    """

    _registries: ClassVar[dict[type, dict[str, type]]] = {}

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> "RegistryMeta":
        """Create class and register it."""
        cls = super().__new__(mcs, name, bases, namespace)

        # Initialize registry for base class
        if not bases:  # This is the base class
            mcs._registries[cls] = {}
        else:
            # Find the base class with registry
            for base in bases:
                if isinstance(base, RegistryMeta):
                    # Get the root registry
                    root = mcs._find_root_registry(base)
                    if root and cls not in mcs._registries.get(root, {}):
                        mcs._registries.setdefault(root, {})[name] = cls
                        logger.debug(f"Registered {name} in {root.__name__} registry")
                    break

        return cls  # type: ignore

    @classmethod
    def _find_root_registry(mcs, cls: type) -> type | None:
        """Find the root class that owns the registry."""
        if cls in mcs._registries:
            return cls
        for base in cls.__bases__:
            if isinstance(base, RegistryMeta):
                root = mcs._find_root_registry(base)
                if root:
                    return root
        return None

    def get_registry(cls) -> dict[str, type]:
        """
        Get all registered subclasses.

        Returns:
            Dictionary mapping class names to class objects.
        """
        root = RegistryMeta._find_root_registry(cls)
        return RegistryMeta._registries.get(root or cls, {})

    def get_class(cls, name: str) -> type | None:
        """
        Get a registered class by name.

        Args:
            name: Name of the class to retrieve.

        Returns:
            The class object or None if not found.
        """
        return cls.get_registry().get(name)

    def create_instance(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Factory method to create an instance of a registered class.

        Args:
            name: Name of the class to instantiate.
            *args: Positional arguments for constructor.
            **kwargs: Keyword arguments for constructor.

        Returns:
            Instance of the registered class.

        Raises:
            KeyError: If class name not found in registry.
        """
        target_cls = cls.get_class(name)
        if target_cls is None:
            raise KeyError(f"Class '{name}' not found in {cls.__name__} registry")

        return target_cls(*args, **kwargs)


class ValidatedMeta(type):
    """
    Metaclass for enforcing class-level validation constraints.

    Validates that classes define required attributes and methods,
    useful for creating strict interfaces without ABC overhead.

    Example:
        >>> class BaseProcessor(metaclass=ValidatedMeta):
        ...     _required_methods = ['process', 'validate']
        ...     _required_attrs = ['name']
        >>> class MyProcessor(BaseProcessor):
        ...     name = "my_processor"
        ...     def process(self): pass
        ...     def validate(self): pass  # Must define all required
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> "ValidatedMeta":
        """Create class and validate requirements."""
        cls = super().__new__(mcs, name, bases, namespace)

        # Skip validation for base classes (those that define requirements)
        if "_required_methods" in namespace or "_required_attrs" in namespace:
            return cls  # type: ignore

        # Collect requirements from base classes
        required_methods: set[str] = set()
        required_attrs: set[str] = set()

        for base in bases:
            if hasattr(base, "_required_methods"):
                required_methods.update(base._required_methods)
            if hasattr(base, "_required_attrs"):
                required_attrs.update(base._required_attrs)

        # Validate required methods
        missing_methods = []
        for method in required_methods:
            if not callable(getattr(cls, method, None)):
                missing_methods.append(method)

        if missing_methods:
            raise TypeError(f"Class {name} missing required methods: {missing_methods}")

        # Validate required attributes
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(cls, attr):
                missing_attrs.append(attr)

        if missing_attrs:
            raise TypeError(
                f"Class {name} missing required attributes: {missing_attrs}"
            )

        logger.debug(f"Validated class {name} successfully")
        return cls  # type: ignore


class PluginMeta(ABCMeta, RegistryMeta):
    """
    Combined metaclass for plugins with ABC and registry support.

    Allows defining abstract base classes that also automatically
    register all concrete implementations.

    Example:
        >>> from abc import abstractmethod
        >>> class Plugin(metaclass=PluginMeta):
        ...     @abstractmethod
        ...     def execute(self): pass
        >>> class ConcretePlugin(Plugin):
        ...     def execute(self):
        ...         return "executed"
        >>> Plugin.get_registry()  # Shows ConcretePlugin
    """

    pass


class CachedPropertyMeta(type):
    """
    Metaclass that converts designated properties to cached properties.

    Automatically wraps properties marked with _cached_properties
    to compute values only once.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> "CachedPropertyMeta":
        """Create class with cached property wrappers."""
        cached_props = namespace.get("_cached_properties", [])

        for prop_name in cached_props:
            if prop_name in namespace:
                original = namespace[prop_name]
                if isinstance(original, property):
                    namespace[prop_name] = mcs._make_cached_property(
                        original.fget, prop_name
                    )

        return super().__new__(mcs, name, bases, namespace)

    @staticmethod
    def _make_cached_property(
        func: Callable[[Any], T] | None,
        name: str,
    ) -> property:
        """Create a cached property from a function."""
        cache_attr = f"_cached_{name}"

        def getter(self: Any) -> T:
            if not hasattr(self, cache_attr):
                if func is None:
                    raise AttributeError(f"No getter for {name}")
                setattr(self, cache_attr, func(self))
            return getattr(self, cache_attr)

        def deleter(self: Any) -> None:
            if hasattr(self, cache_attr):
                delattr(self, cache_attr)

        return property(getter, None, deleter)


# Utility function for inheritance checking
def ensure_base_class(base: type) -> Callable[[type], type]:
    """
    Class decorator to ensure a class inherits from specified base.

    Args:
        base: Required base class.

    Returns:
        Decorator function.
    """

    def decorator(cls: type) -> type:
        if not issubclass(cls, base):
            raise TypeError(f"Class {cls.__name__} must inherit from {base.__name__}")
        return cls

    return decorator


# Module testing
if __name__ == "__main__":
    # Test SingletonMeta
    class Config(metaclass=SingletonMeta):
        def __init__(self, env: str = "dev"):
            self.env = env

    c1 = Config("production")
    c2 = Config("staging")
    print(f"Singleton test: c1 is c2 = {c1 is c2}")
    print(f"Environment: {c1.env}")  # Will be "production"

    # Test RegistryMeta
    class Transformer(metaclass=RegistryMeta):
        pass

    class CleanseTransformer(Transformer):
        pass

    class ValidateTransformer(Transformer):
        pass

    print(f"Registry: {Transformer.get_registry()}")

    # Test ValidatedMeta
    class BaseService(metaclass=ValidatedMeta):
        _required_methods = ["execute"]
        _required_attrs = ["name"]

    try:

        class BadService(BaseService):
            pass  # Missing required items

    except TypeError as e:
        print(f"Validation error (expected): {e}")

    class GoodService(BaseService):
        name = "good_service"

        def execute(self):
            return "done"

    print(f"GoodService.name = {GoodService.name}")
