import inspect

from autofit.tools.util import get_class_path, get_class


class Dictable:
    def dict(self) -> dict:
        """
        A dictionary representation of the instance comprising a type
        field which contains the entire class path by which the type
        can be imported and constructor arguments.
        """
        argument_dict = {
            arg: getattr(
                self, arg
            )
            for arg
            in inspect.getfullargspec(
                self.__init__
            ).args[1:]
        }
        return {
            "type": get_class_path(
                self.__class__
            ),
            **argument_dict
        }

    @staticmethod
    def from_dict(
            profile_dict
    ):
        """
        Instantiate a GeometryProfile from its dictionary representation.

        Parameters
        ----------
        profile_dict
            A dictionary representation of the instance comprising a type
            field which contains the entire class path by which the type
            can be imported and constructor arguments.

        Returns
        -------
        An instance of the geometry profile specified by the type field in
        the profile_dict
        """
        cls = get_class(
            profile_dict.pop(
                "type"
            )
        )
        # noinspection PyArgumentList
        return cls(
            **profile_dict
        )
