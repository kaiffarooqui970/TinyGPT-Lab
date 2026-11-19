# Feature labels — S5 layer-3 TopK k=32 SAE, 50 random latents (seed 0)

**Status: DRAFT by Claude, pending review by the author.** Labels are based on the top-activating contexts and logit effects in `report/dashboard/features.html`.

Summary: 37 interpretable, 8 partly, 5 not (of 50).

| feature | label | description |
|---|---|---|
| 16 | not | Position-0 artifact: fires at ~15.1 on the first token of each 256-token window, whatever the token |
| 32 | interpretable | Sub-word 'cur' (curved, cursed, curling) |
| 50 | interpretable | Blank line after '…?" asked X.', i.e. the start of the reply turn in a dialogue |
| 100 | interpretable | <|endoftext|> story boundary |
| 135 | interpretable | 'and' joining items in a list of nouns (plates and cups, chairs and tables) |
| 173 | partly | 'wheels' after 'big'/'four big'; also fires on 'walls' |
| 205 | partly | Weak: 'a' introducing a new entity ('there was a', 'found a', 'saw a') |
| 249 | interpretable | 'anymore' after 'not scared/grumpy' |
| 459 | interpretable | 'joy' in 'with joy' / 'in joy' |
| 492 | interpretable | 'listened' ('I should have listened to you') |
| 547 | partly | Sentence-final period, mostly after habitual clauses ('every day.') |
| 762 | interpretable | First letter 'J' of a character name opening a new story |
| 1069 | interpretable | 'told' after a parent reacts ('mom got upset and told him…') |
| 1076 | interpretable | 'of' in 'in the middle of' |
| 1580 | interpretable | Closing punctuation inside a quoted utterance, just before the closing quote |
| 1645 | interpretable | 'such' in 'such a [adjective] [noun]' |
| 1695 | interpretable | 'showed/showing' (showing something off to someone) |
| 1837 | interpretable | Closing quote after a quoted polite word ('please', 'thank you', 'no') |
| 1877 | interpretable | 'went' after a stated want ('he was hungry… so he went to') |
| 2411 | interpretable | Opening quote of a reply line that answers a question |
| 2473 | partly | Adjective after sentence-initial 'The' ('The old man', 'The nice man') |
| 2592 | partly | Sub-word 'n' (nook, numb, natural), plus 'other' |
| 2950 | interpretable | Shape nouns ('circle', 'square') |
| 3075 | interpretable | 'right' in 'the right [thing/words/time]' |
| 3116 | interpretable | 'lion' |
| 3227 | interpretable | 'living' in 'living room' |
| 3319 | interpretable | Mojibake '€' of a curly opening quote at the start of a reply line (data-encoding feature) |
| 3322 | interpretable | 'brush' (hair, teeth, paint) |
| 3392 | interpretable | 'home' in 'When they got home,' |
| 3422 | interpretable | 'lot' in 'a lot' |
| 3705 | interpretable | Sentence-initial 'There' introducing a discovery ('There she saw…') |
| 3780 | interpretable | 'a' in the story opener 'Once upon a time, there was a' |
| 3863 | not | Position-0 artifact: 'They' at the first token of a window, constant activation 9.86 |
| 3882 | interpretable | Paragraph-initial 'On' ('On the way', 'On their way home') |
| 3964 | interpretable | 'thirsty' |
| 3975 | interpretable | 'nice' in the moral 'it is not nice to…' |
| 4103 | not | Position-0 artifact: "'s" at the first token of a window, constant activation 12.18 |
| 4117 | partly | 'the' after a motion verb toward a place ('arrived at the', 'explore the'); frequent (3.4%) |
| 4457 | interpretable | Intensifier 'so' after 'was/were' ('was so happy') |
| 4468 | interpretable | 'baby' ('a mommy and a baby', 'baby bird') |
| 4683 | not | Weak and mixed ('me', 'now') with no clear pattern |
| 4964 | interpretable | 'pay' |
| 4989 | interpretable | 'sun' in 'until the sun went down/set' |
| 5184 | interpretable | 'fight/fighting' |
| 5185 | interpretable | Preposition after 'thought' ('thought of', 'thought about') |
| 5247 | interpretable | Suffix token 'ed' split off a word (erupt-ed, address-ed) |
| 5289 | partly | 'a' after 'in' ('in a big house', 'in a forest') |
| 5573 | partly | 'had' after a subject; several top contexts are near the window start |
| 5717 | not | Mixed mid-word pieces (transf-orm, prod-uc) and 'of' |
| 5930 | interpretable | 'every' in '(played together) every day' |
