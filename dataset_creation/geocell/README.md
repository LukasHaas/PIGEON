# Geocell Creation

This folder contains the following code files. The geocell creation algorithm is very complex and a major contribution of our work.

## ```cell.py```

Implementation of a geocell abstraction. Includes logic to Voronoi tesselate, combine, and split geocells. A geocell consists of both points (training samples) and a set of shapes, the union of which define its geographic area.

## ```cell_collection.py```

Inherits from Phython's ```set``` class and is combines multiple geocells into a set to build a hierarchy of geocells. Contains code to generate geocells efficiently in a multi-CPU setup.

## ```geocell_creation.py```

Initializes the data used to create a geocell collection and starts the geocell creation algorithm.

## ```naive_cell.py```

Implementation of our naive geocell creation algorithm.

## ```voronoi.py```

Python implementation of Voronoi tesselation.
