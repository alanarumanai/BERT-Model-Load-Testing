"""Sample review texts for load testing, grouped by length.

Lengths are chosen to exercise the model's cost range (max_length is 128 tokens):
  short:  about 8-15 tokens
  medium: about 35-50 tokens
  long:   about 110-128 tokens, close to the truncation limit
Each group mixes positive and negative reviews.
"""

SHORT = [
    "A wonderful film from start to finish.",
    "Dull, predictable and far too long.",
    "The cast was brilliant and the music unforgettable.",
    "I walked out halfway through.",
    "Easily the best thing I have seen all year.",
    "The jokes fell flat every single time.",
    "Charming, funny and surprisingly moving.",
    "A confusing mess with no clear story.",
]

MEDIUM = [
    (
        "I expected very little from this film, but the writing was sharp, the lead actor "
        "gave a warm and believable performance, and the final scene left the whole "
        "audience smiling."
    ),
    (
        "The trailer promised an exciting thriller, yet the film drags through slow scenes, "
        "repeats the same twist twice and ends without explaining any of the questions it "
        "raised."
    ),
    (
        "Beautifully shot on location, with a gentle pace that suits the story, this is a "
        "film that trusts its audience and rewards patience with a genuinely touching "
        "ending."
    ),
    (
        "Despite a talented cast, the dialogue feels forced, the characters make choices "
        "that make no sense, and the loud soundtrack tries too hard to create tension that "
        "is never there."
    ),
    (
        "My children loved the colourful animation and silly songs, and to my surprise I "
        "laughed just as much as they did, especially at the clever jokes aimed at adults."
    ),
    (
        "The special effects are impressive, but they cannot hide a thin plot, flat "
        "performances and a running time that is at least forty minutes longer than it "
        "needed to be."
    ),
]

LONG = [
    (
        "I went into this film knowing almost nothing about it, and I came out convinced it "
        "was one of the most thoughtful dramas of the decade. The story follows a retired "
        "teacher who returns to her childhood village to care for her brother, and the "
        "script handles their difficult history with real patience and honesty. Every "
        "supporting character feels like a real person with their own worries, the "
        "photography makes the quiet countryside look both lovely and lonely, and the "
        "restrained score never tells you what to feel. When the final scene arrived, "
        "several people around me were quietly crying, and so was I."
    ),
    (
        "It is hard to remember the last time a big studio film wasted so much talent. The "
        "opening twenty minutes are promising, with a clever setup and a likeable hero, but "
        "after that the story collapses into a series of noisy chase scenes that have no "
        "real consequences. Characters appear and disappear without explanation, the "
        "villain's plan changes from one scene to the next, and the dialogue is so full of "
        "clumsy exposition that the actors seem embarrassed to say it. By the time the "
        "endless final battle began, I had stopped caring about anyone on the screen and "
        "simply wanted to go home."
    ),
    (
        "This documentary about a small orchestra in a struggling factory town is a genuine "
        "delight. The director spends a full year with the musicians, most of whom have "
        "full-time jobs and rehearse late at night, and lets them tell their own stories "
        "without any narration getting in the way. We see arguments, injuries, lost "
        "funding and a disastrous first concert, but also friendship, humour and an "
        "astonishing amount of determination. The final performance is filmed with such "
        "care that you feel you are sitting in the hall with the families who came to "
        "watch, and it is impossible not to cheer for them."
    ),
    (
        "I really wanted to like this adaptation, because the novel it is based on is one of "
        "my favourite books, but almost everything that made the original special has been "
        "removed. The careful, slow-building mystery has been replaced by cheap jump "
        "scares, the complex narrator has become a bland action hero, and the thoughtful "
        "ending has been rewritten into something predictable and sentimental. The "
        "production design is attractive and one or two supporting performances are "
        "strong, but they cannot rescue a script that seems to misunderstand its own "
        "source material at every turn."
    ),
]

TEXTS = {"short": SHORT, "medium": MEDIUM, "long": LONG}
