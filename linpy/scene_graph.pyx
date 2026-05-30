# scene_graph.pyx — Cython implementation of SceneGraph
from .transform cimport Transform
from .vector3 cimport Vector3
from .quaternion cimport Quaternion

def print_tree(int depth, Transform transform):
    """Recursively print the transform hierarchy as an indented tree.

    :param depth: Current indentation depth.
    :param transform: The transform node to print.
    """
    print(depth * ' ' + transform.name)
    cdef Transform child
    for child in transform:
        print_tree(depth + 1, child)


cdef class SceneGraph:
    """A hierarchical scene graph that manages named transforms in a tree structure."""

    root_name = "__root__"

    def __cinit__(self):
        self._transforms = {}
        self._root = Transform(Vector3(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0), self.root_name)

    def __init__(self):
        ...

    def __getitem__(self, str key) -> Transform:
        """Retrieve a transform by name.

        :param key: The name of the transform.
        :return: The transform associated with the given name.
        :raises KeyError: If the name is not found.
        """
        return self._transforms[key]

    def __contains__(self, item):
        return item in self._transforms

    def __iter__(self):
        return iter(self._transforms)

    def __len__(self):
        return len(self._transforms)

    @property
    def root(self) -> Transform:
        """The root transform of the scene graph."""
        return self._root

    cpdef void apply_transform(self, str transform_name, str parent_name, Vector3 local_position, Quaternion local_rotation):
        """Add or update a named transform in the scene graph.

        If the transform already exists, its local position, rotation, and parent
        are updated. Otherwise a new transform is created under the specified parent.
        If the parent does not yet exist, a placeholder is created under root.

        :param transform_name: Name of the transform to add or update.
        :param parent_name: Name of the parent transform.
        :param local_position: Local position relative to the parent.
        :param local_rotation: Local rotation relative to the parent.
        """
        self.c_apply_transform(transform_name, parent_name, local_position, local_rotation)

    cpdef void remove(self, str transform_name):
        """Remove a transform from the scene graph by name.

        Children of the removed transform are reparented to its parent.
        Does nothing if the name is not found.

        :param transform_name: Name of the transform to remove.
        """
        self.c_remove(transform_name)

    cpdef void print_graph(self):
        """Print the entire scene graph hierarchy to stdout."""
        print_tree(0, self._root)



    cdef void c_apply_transform(self, str transform_name, str parent_name, Vector3 local_position, Quaternion local_rotation):
        cdef Transform parent
        cdef Transform t

        # Resolve or create the parent
        if parent_name == self.root_name:
            parent = self._root
        elif parent_name in self._transforms:
            parent = <Transform>self._transforms[parent_name]
        else:
            # Parent not seen yet — create a placeholder under root
            parent = Transform(Vector3(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0), parent_name)
            self._root.c_add_child(parent)
            self._transforms[parent_name] = parent

        if transform_name in self._transforms:
            t = <Transform>self._transforms[transform_name]

            if t._parent is not parent:
                t.c_set_local_pos_rot_parent(local_position, local_rotation, parent)
            else:
                t.c_set_local_pos_rot(local_position, local_rotation)
        else:
            # Brand-new transform
            t = Transform(local_position, local_rotation, transform_name)
            parent.c_add_child(t)
            self._transforms[transform_name] = t

    cdef void c_remove(self, str transform_name):
        if transform_name not in self._transforms:
            return

        cdef Transform t = <Transform>self._transforms.pop(transform_name)
        cdef Transform parent_t = t._parent
        cdef Transform child

        for child in list(t):
            child.parent = parent_t
        t.parent = None
