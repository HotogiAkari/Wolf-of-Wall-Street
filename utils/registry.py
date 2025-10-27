# 文件路径: utils/registry.py

from collections import defaultdict

class Registry:
    """
    通用的,支持phase和requires的注册器类。
    """
    def __init__(self):
        # 按阶段存储注册的项
        self._registry = defaultdict(list)
        # 存储每个项的详细信息（包括它需要什么）
        self._item_details = {}

    def register(self, name: str, phase: str = 'default', requires: list = None):
        """
        一个装饰器，用于注册一个类或函数。

        :param name: 注册项的唯一名称。
        :param phase: 该项所属的执行阶段 (e.g., 'base', 'contextual', 'final')。
        :param requires: 一个列表，包含该项运行前必须存在的特征（列名）。
        """
        def wrapper(cls):
            if name in self._item_details:
                raise ValueError(f"'{name}' is already registered.")
            
            self._registry[phase].append(cls)
            self._item_details[name] = {
                'class': cls,
                'phase': phase,
                'requires': set(requires or [])
            }
            # 将注册名附加到类上，方便内部访问
            cls._registry_name = name
            return cls
        return wrapper

    def get_registered_items(self, phases: list) -> list:
        """
        按照指定阶段的顺序，返回所有已注册的类。
        未来可以扩展此方法以执行拓扑排序来处理更复杂的依赖。
        """
        items = []
        for phase in phases:
            items.extend(self._registry.get(phase, []))
        return items
    
class SimpleRegistry:
    """
    简单的,用于模型的注册器类
    """
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def wrapper(cls):
            if name in self._registry:
                raise ValueError(f"'{name}' is already registered in SimpleRegistry.")
            self._registry[name] = cls
            return cls
        return wrapper

    def get(self, name):
        cls = self._registry.get(name)
        if cls is None:
            raise ValueError(f"'{name}' is not a registered item. Available: {list(self._registry.keys())}")
        return cls