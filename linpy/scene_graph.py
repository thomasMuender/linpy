from .transform import Transform
from .vector import Vector3
from .quaternion import Quaternion


def print_tree(depth: int, transform: Transform) -> None:
    print(depth * ' ' + transform.name)
    for child in transform:
        print_tree(depth + 1, child)


class SceneGraph():

    root_name: str = "__root__"

    def __init__(self) -> None:
        self.__transforms: dict[str, Transform] = {}
        self.__root = Transform(Vector3.zero(), Quaternion.identity(), self.root_name)


    def __getitem__(self, key):
        return self.__transforms.__getitem__(key)


    @property
    def root(self) -> Transform:
        return self.__root

    
    def apply_transform(self, transform_name: str, parent_name: str, local_position: Vector3, local_rotation: Quaternion) -> None:
        # Resolve or create the parent
        if parent_name == self.root_name:
            parent = self.__root
        elif parent_name in self.__transforms:
            parent = self.__transforms[parent_name]
        else:
            # Parent not seen yet — create a placeholder under root
            parent = Transform(Vector3.zero(), Quaternion.identity(), parent_name)
            self.__root.add_child(parent)
            self.__transforms[parent_name] = parent

        if transform_name in self.__transforms:
            t = self.__transforms[transform_name]
            t.local_position = local_position
            t.local_rotation = local_rotation
            if t.parent is not parent:
                t.parent = parent
        else:
            # Brand-new transform
            t = Transform(local_position, local_rotation, transform_name)
            parent.add_child(t)
            self.__transforms[transform_name] = t

    
    def print_graph(self) -> None:
        print_tree(0, self.__root)


    def remove(self, transform_name: str) -> None:
        if transform_name not in self.__transforms:
            return
        t = self.__transforms.pop(transform_name)
        # Reparent children to the removed node's parent
        parent = t.parent
        for child in list(t):
            child.parent = parent
        t.parent = None