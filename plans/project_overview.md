# Goal
The goal is to create a flashcard generation system.

The general process is as follows:
1. Read a book / article and create highlights
2. [Extract the core information](#extract-the-core-information) from the highlights into
3. For each highlight [create flashcards](#create-flashcards)

# Additional Context
## Extract the core information
I'd like to parse all highlights and figure out what core concepts are covered.
The goal of this step is to create a mental model of the highlights.

I generally do the following to create a knowledge graph:
For each highlight:
1. Read the highlight and reread any surrounding context from the original source material.
2. Figure out the core information in the highlight. There may be multiple concepts within one highlight. Many highlights may share similar information.
3. For each concept, see if there's already a concept / node in the knowledge graph. If the node doesn't doesn't exist, create a new one and then see if it connects to any other nodes.
4. Repeat this until all highlights are processed.

To see an example, the [highlights for "How We Learn"](../../examples/How%20We%20Learn%20-%20Benedict%20Carey.md) creates [this knowledge graph](../../examples/How%20We%20Learn%20-%20Benedict%20Carey.canvas)

## Create Flashcards
Given the above knowledge graph, I want to create flashcards that help encode the knowledge graph information.

I generally do the following to create flashcards from the knowledge graph:
1. For node / concept create a flashcard that encodes information. I use the following article as a reference for best practices: https://andymatuschak.org/prompts/.