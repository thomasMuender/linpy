from .transform import Transform
from .vector import Vector3
from .quaternion import Quaternion

def print_tree(depth: int, transform: Transform) -> None:
    """Recursively print the transform hierarchy as an indented tree.

    :param depth: Current indentation depth.
    :param transform: The transform node to print.
    """
    print(depth * ' ' + transform.name)
    for child in transform:
        print_tree(depth + 1, child)

class SceneGraph():
    """A hierarchical scene graph that manages named transforms in a tree structure."""

    root_name: str = "__root__"

    def __init__(self) -> None:
        """Initialize an empty scene graph with a root transform."""
        self.__transforms: dict[str, Transform] = {}
        self.__root = Transform(Vector3.zero, Quaternion.identity, self.root_name)

    def __getitem__(self, key: str) -> Transform:
        """Retrieve a transform by name.

        :param key: The name of the transform.
        :return: The transform associated with the given name.
        :raises KeyError: If the name is not found.
        """
        return self.__transforms.__getitem__(key)

    @property
    def root(self) -> Transform:
        """The root transform of the scene graph.

        :return: The root transform.
        :rtype: Transform
        """
        return self.__root

    def apply_transform(self, transform_name: str, parent_name: str, local_position: Vector3, local_rotation: Quaternion) -> None:
        """Add or update a named transform in the scene graph.

        If the transform already exists, its local position, rotation, and parent
        are updated. Otherwise a new transform is created under the specified parent.
        If the parent does not yet exist, a placeholder is created under root.

        :param transform_name: Name of the transform to add or update.
        :param parent_name: Name of the parent transform.
        :param local_position: Local position relative to the parent.
        :param local_rotation: Local rotation relative to the parent.
        """
        # Resolve or create the parent
        if parent_name == self.root_name:
            parent = self.__root
        elif parent_name in self.__transforms:
            parent = self.__transforms[parent_name]
        else:
            # Parent not seen yet — create a placeholder under root
            parent = Transform(Vector3.zero, Quaternion.identity, parent_name)
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
        """Print the entire scene graph hierarchy to stdout."""
        print_tree(0, self.__root)

    def remove(self, transform_name: str) -> None:
        """Remove a transform from the scene graph by name.

        Children of the removed transform are reparented to its parent.
        Does nothing if the name is not found.

        :param transform_name: Name of the transform to remove.
        """
        if transform_name not in self.__transforms:
            return
        t = self.__transforms.pop(transform_name)
        # Reparent children to the removed node's parent
        parent = t.parent
        for child in list(t):
            child.parent = parent
        t.parent = None