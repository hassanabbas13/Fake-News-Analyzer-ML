# REAL TESTING

Ten articles for testing the reading model, five real and five fake.

Every one of these is **written from scratch**, so none of them is in the
database. That is the point: Step 1 has to miss, so the model is forced to
answer. Paste one into Analyze and the result page should say
**Reading model**, never **Found in our dataset**.

Paste the whole block, headline and body together. The first line is read as
the headline and the rest as the article.

## What to expect

| | Verdict | Shown confidence |
|---|---|---|
| the five REAL | Likely Real | 90.1% |
| the five FAKE | Likely Fake | 86.9% |

Those two percentages are fixed for every article of that kind. They are not
how sure the model is about your specific paste; they are how often a verdict
of that kind turned out to be right, measured on 3,160 articles the
model never trained on. So all five real ones show the same number, and so do
all five fake ones. That is correct, not a bug.

No web search should run on any of these. All ten score outside the unsure
band, so the model answers alone.

---

## The five REAL ones

Plain wire-service reporting: attribution, figures, hedges, no adjectives
doing emotional work. This is what the model learned "real" looks like.

### REAL 1 — Central bank holds interest rates steady at four percent

*102 words. Model score 0.0002. Expect **Likely Real**.*

```
Central bank holds interest rates steady at four percent

The central bank left its benchmark interest rate unchanged at 4 percent on Thursday, citing easing inflation and a cooling labour market. In a statement following the two-day meeting, policymakers said they would continue to assess incoming data before adjusting policy further. Eight of the nine committee members voted to hold, with one favouring a quarter-point cut. The decision was in line with expectations from economists surveyed last week. Consumer price growth slowed to 2.4 percent in August from 2.7 percent in July, according to figures published by the statistics office.
```

### REAL 2 — Manufacturing output rose 0.3 percent in August, statistics office says

*101 words. Model score 0.0002. Expect **Likely Real**.*

```
Manufacturing output rose 0.3 percent in August, statistics office says

Industrial production increased 0.3 percent in August compared with the previous month, the national statistics office said on Wednesday, slightly above the 0.2 percent forecast by analysts. Output in the automotive sector rose 1.1 percent, while chemical production fell 0.4 percent. On an annual basis, manufacturing output was 1.8 percent higher than in the same month a year earlier. A spokesperson for the office said the figures were subject to revision. Separate data released on Tuesday showed factory orders climbing for a third consecutive month.
```

### REAL 3 — Court adjourns hearing in transport funding case until October

*101 words. Model score 0.0003. Expect **Likely Real**.*

```
Court adjourns hearing in transport funding case until October

A federal appeals court adjourned proceedings on Tuesday in a dispute over the allocation of regional transport funding, setting the next hearing for 14 October. Lawyers for the two municipalities involved argued that the funding formula did not account for population changes recorded in the most recent census. Counsel for the transport ministry said the formula had been applied consistently and was reviewed every four years. The three-judge panel asked both sides to submit written arguments by the end of the month. The court did not indicate when it expected to rule.
```

### REAL 4 — Health ministry reports decline in seasonal flu cases

*93 words. Model score 0.0002. Expect **Likely Real**.*

```
Health ministry reports decline in seasonal flu cases

Reported cases of seasonal influenza fell for a fourth consecutive week, the health ministry said in its weekly surveillance bulletin published on Friday. Laboratories confirmed 1,842 cases in the week to 30 August, down from 2,103 the previous week. Hospital admissions related to respiratory illness declined 6 percent over the same period. The ministry said vaccination coverage among adults over 65 stood at 61 percent, roughly unchanged from last year. Officials encouraged eligible residents to receive the vaccine ahead of the winter months.
```

### REAL 5 — Two logistics companies agree merger terms in three billion euro deal

*101 words. Model score 0.0002. Expect **Likely Real**.*

```
Two logistics companies agree merger terms in three billion euro deal

Two logistics companies said on Monday they had agreed terms for a merger valuing the combined group at about 3 billion euros. Under the agreement, shareholders in the smaller firm will receive 0.42 shares in the acquirer for each share held. The boards of both companies unanimously recommended the transaction, which remains subject to regulatory approval and a shareholder vote expected in the first quarter. The companies said they anticipated annual cost savings of 90 million euros by the third year. Shares in the target closed 11 percent higher.
```

---

## The five FAKE ones

Hyperpartisan clickbait: capitals mid-sentence, second person, an enemy who
is hiding something, an instruction to share. No checkable facts anywhere.

### FAKE 1 — BREAKING: Insider LEAKS The Document They Never Wanted You To See (VIDEO)

*108 words. Model score 0.9999. Expect **Likely Fake**.*

```
BREAKING: Insider LEAKS The Document They Never Wanted You To See (VIDEO)

You are not going to believe what just came out. A whistleblower deep inside the agency has finally come forward with the documents that prove everything we have been saying for years, and the mainstream media is already scrambling to bury the story. WATCH the video below before they take it down. These crooked politicians thought they could keep this hidden forever, but the truth always comes out in the end. Every single one of them needs to be held accountable, and the American people deserve answers RIGHT NOW. Share this everywhere before it gets censored.
```

### FAKE 2 — Liberals MELT DOWN After Patriot DESTROYS Them With One Simple Fact

*108 words. Model score 0.9998. Expect **Likely Fake**.*

```
Liberals MELT DOWN After Patriot DESTROYS Them With One Simple Fact

It was absolutely glorious. The entire panel sat there in stunned silence, unable to respond, as one brave patriot laid out the facts they have spent years trying to hide from you. The look on their faces was priceless and the internet is going wild. This is exactly what happens when you confront these people with reality instead of their talking points. The liberal establishment does not want you to see this clip, which is why they are already trying to get it removed from every platform. Watch it below and share it with everyone you know.
```

### FAKE 3 — Doctors STUNNED: This Common Kitchen Item Reverses Aging Overnight

*103 words. Model score 0.9999. Expect **Likely Fake**.*

```
Doctors STUNNED: This Common Kitchen Item Reverses Aging Overnight

Big Pharma has been hiding this simple secret for decades because there is no money in a cure. Researchers were shocked when they discovered that this ordinary item sitting in your kitchen right now can reverse the aging process almost overnight, and thousands of people are already seeing unbelievable results. The medical establishment refuses to talk about it, and one doctor who tried to speak out was silenced immediately. You will not hear about this on the evening news. Click here to learn the shocking truth they do not want you to know about.
```

### FAKE 4 — EXPOSED: Secret Meeting Proves The Whole Thing Was RIGGED From Day One

*105 words. Model score 0.9999. Expect **Likely Fake**.*

```
EXPOSED: Secret Meeting Proves The Whole Thing Was RIGGED From Day One

The evidence is now overwhelming and undeniable. A secret meeting held behind closed doors proves beyond any doubt that the entire process was rigged from the very beginning, exactly as patriots warned. The corrupt establishment media will never report on this because they were in on it the whole time. Anyone who dares question the official narrative gets smeared and censored instantly. This is the biggest scandal in the history of our nation and not one single person has been arrested. Wake up America, before it is too late for all of us.
```

### FAKE 5 — She Said WHAT?! Shocking Hot Mic Moment Ends Her Career Instantly

*107 words. Model score 0.9998. Expect **Likely Fake**.*

```
She Said WHAT?! Shocking Hot Mic Moment Ends Her Career Instantly

She had no idea the microphone was still on, and what she said next left everyone in the room absolutely speechless. Her career is officially over and she has nobody to blame but herself. The video is spreading like wildfire across social media and her team is in full panic mode trying to do damage control. Of course, the mainstream media is completely silent about the whole thing, because they always protect their own. This is who these people really are when they think nobody is listening. Watch the unbelievable footage below and decide for yourself.
```

---

## Notes

**The model reads style, not truth.** It has no idea whether a central bank
met on Thursday. It recognises the shape of wire copy and the shape of
clickbait. Worth knowing when you read a verdict: a lie told in careful prose
is the case it handles worst.

**Why there are no borderline cases here.** The scores below are all past
0.999 or under 0.001, with nothing in between, because the model is close to
binary on style. Getting one to land in the middle took a sweep of the same
story written nine different ways.

| id | score | verdict |
|---|---|---|
| R1 | 0.0002 | Likely Real |
| R2 | 0.0002 | Likely Real |
| R3 | 0.0003 | Likely Real |
| R4 | 0.0002 | Likely Real |
| R5 | 0.0002 | Likely Real |
| F1 | 0.9999 | Likely Fake |
| F2 | 0.9998 | Likely Fake |
| F3 | 0.9999 | Likely Fake |
| F4 | 0.9999 | Likely Fake |
| F5 | 0.9998 | Likely Fake |

The cutoff is 0.4. Above it the answer is Likely Fake, below it Likely Real.

**If one of these ever returns "Found in our dataset"**, it means someone has
since loaded a collection containing it, and the article is no longer testing
what this file says it tests.

