"""Built-in sample paper (used for the demo, tests and the sample PDF)."""

ABSTRACT = ("We present a lightweight framework that summarizes research papers using only their introduction and "
            "conclusion. Sentences are embedded with a pre-trained BERT encoder and the most representative ones are "
            "selected by clustering. Experiments on computer science papers show that summaries are more than sixty "
            "percent shorter while keeping the research problem, approach and findings.")

INTRO_1 = ("The volume of scientific literature is growing at an unprecedented rate, with millions of new research "
           "papers published every year. University repositories now contain lengthy documents that make it difficult "
           "for students and researchers to quickly understand the main contribution of a paper. Reading full papers "
           "to decide whether they are relevant is time consuming and often impractical. Automatic text summarization "
           "has therefore become an important tool for helping readers navigate this growing body of knowledge.")

INTRO_2 = ("Traditional summarization techniques rely on hand-crafted features such as word frequency and sentence "
           "position, which fail to capture the deeper semantic meaning of scientific text. Recent advances in deep "
           "learning, and in particular transformer-based language models such as BERT, have shown a remarkable "
           "ability to represent the meaning of sentences in context. However, most existing summarization systems are "
           "trained on news articles and do not transfer well to the specialised vocabulary and structure of academic "
           "writing.")

INTRO_3 = ("In this paper, we propose a summarization approach that uses a pre-trained BERT model to encode the "
           "sentences of a paper and selects the most representative ones using clustering. We focus on the "
           "introduction and the conclusion because these sections state the research problem, the motivation, and the "
           "final outcomes of the work. Our goal is to produce short summaries that preserve the key research "
           "information without requiring any task-specific training data.")

CONCLUSION = ("In this work, we presented a BERT-based framework for summarizing research papers using only their "
              "introduction and conclusion. Our method embeds every sentence with a pre-trained BERT encoder and "
              "applies K-Means clustering to select the sentences closest to each cluster centre. Experiments on a "
              "collection of computer science papers show that the generated summaries reduce the text length by more "
              "than sixty percent while retaining the main research problem, approach, and findings. The results "
              "demonstrate that contextual embeddings capture the semantic structure of academic text far better than "
              "frequency-based methods. We also observed that combining information from both the introduction and the "
              "conclusion yields more complete summaries than using either section alone. Nevertheless, the extractive "
              "nature of the method means that some summaries can contain sentences that lack context when read in "
              "isolation. In future work, we plan to fine-tune the model on scientific corpora and to explore "
              "abstractive generation to produce more fluent summaries. Overall, we conclude that pre-trained language "
              "models provide a practical and effective solution for helping readers quickly grasp the contribution "
              "of a paper.")

SAMPLE_INTRO = " ".join([INTRO_1, INTRO_2, INTRO_3])
SAMPLE_CONCLUSION = CONCLUSION
SAMPLE_ABSTRACT = ABSTRACT
