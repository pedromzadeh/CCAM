from setuptools import setup

setup(
    name="collider",
    version="1.0",
    author="Pedrom Zadeh",
    description="2D cell collisions simulated within phase-field framework",
    packages=[
        "box",
        "cell",
        "polarity",
        "potential",
        "simulator",
        "substrate",
        "helper_functions",
        "visuals",
    ]
)
