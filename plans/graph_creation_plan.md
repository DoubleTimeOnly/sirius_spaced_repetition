# Goal
Given a set of clusters and their highlights (see `src/sirius/protocols.py`), create a visual knowledge graph relating all the information together.

See the [project overview's](project_overview.md#extract-the-core-information). The current pipeline already groups highlights into semantically similar groups. This new feature will take these clusters, create nodes in the knowledge graph, and find relevant connections between those nodes.

# Example
See the `examples` folder
* [The highlights](../examples/How%20We%20Learn%20-%20Benedict%20Carey.md)
* [The knowledge graph](../examples/How%20We%20Learn%20-%20Benedict%20Carey.canvas)

The given highlights produced the given knowledge graph. Note the level of information compression and types of connections made.
This is not a strict rule, but merely an example to push you in the right direction. You don't have to follow the example exactly, except in the format of the knowlegdge graph.

# Format
The knowledge graph is in Obsidian's [JSON Canvas](https://jsoncanvas.org/spec/1.0/) format.

# Implementation details
This should be added into the current pipeline configuration. Update `pipeline.py` to have a new step after clustering highlights, which takes in a ClusterMapping and outputs a knowledge in JSON canvas format.
Additionally the knowledge graph should be saved to disk. When running the pipeline create a folder in the format `YYYY-MM-DD:HH-MM-{highlight_file_name}`. In that folder save a file called `knowledge-graph-{highlight_file_name}.canvas`.
