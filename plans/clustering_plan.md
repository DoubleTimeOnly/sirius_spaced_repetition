# Goal
For an overview, see the [project overview](./project_overview.md). As an initial experiment, I'd like to see if I can reliably cluster semantically similar highlights to make knowledge graph formation easier.

Given a set of highlights, I'd like to group them based on their content, where the output is a set of clusters and the highlights they contain.

# The idea
For each highlight
1. Break down the highlight into its core information. This should be implemented as a function that takes in a string representing the highlight (and optional surrounding context) and outputs a string representing the core information. This should be abstracted so the underlying encoding method can be changed and easily experimented with.
2. Given a string representing the core information, encode it into a latent vector.
3. Cluster all of the encoded vectors. The clustering function should take a set of latent vectors (representing highlights) and return a dictionary of key = cluster and value = set of vectors idxs that belong to that cluster. Importantly, each vector (highlight) can potentially belong to multiple clusters as highlights may contain multiple ideas / concepts. Once again this design should be abstracted such that the underlying clustering method can easily be changed.

So overall the input is a set of highlights and the output is a mapping of clusters and their highlights.
