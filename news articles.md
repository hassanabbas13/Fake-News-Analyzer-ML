# News Articles for Testing

Twenty real articles pulled straight out of the app's own database
(59,660 rows). Paste them into the Analyze page to see how each of the three
steps behaves.

The article text is the stored extract — about 100 words — so every one of
these ends mid-sentence. That is normal; the database only keeps an opening
chunk, not the full body.

---

## Part 1 — Headline + article

Paste the whole block, headline included. All ten are confirmed to hit the
database, so you should get **Database match** and **100%** every time. The
model is never consulted.

### 1. REAL — All the News 2.1 (15 newsrooms)

```
On Mormon spring break, hooking up means getting hitched

On Memorial Day weekend, hundreds of single Mormons in their 20s and 30s converged at Duck Beach on North Carolina’s Outer Banks for their version of spring break. But they don’t go to party or hook up. Many of them are on a mission: to marry. This year, VICE News joined the annual gathering of singles. Most of the LDS Church beachgoers we met were working professionals from the D.C. area who appreciated the opportunity to expand their dating prospects. They’re OK with the fact that they’re taking longer to settle down than Utah-based Mormons, but they still feel strong pressure from the commu...
```

Expected: **Likely Real** · Database · 100%

### 2. FAKE — Kaggle ISOT

```
Hillary UNLOADS On Trump For Attacking Human Rights, Shows GOP Who SHOULD Have Been POTUS (VIDEO)

On Thursday, Hillary Clinton ripped Donald Trump and his bigoted administration apart for being on the wrong side of history and refusing to defend LGBT rights   and while she was at it, she showed conservatives just what they missed out on by letting a former reality television star get into the White House instead of a well-qualified politician.At a fundraiser for LGBT community organization The Center, Clinton received an award and gave a speech in which she thanked her audience for their continued support. However, her message came with a chilling warning about the Trump administration as ...
```

Expected: **Likely Fake** · Database · 100%

### 3. REAL — All the News 2.1 (15 newsrooms)

```
Mountain Rock Kings Karma to Burn are Still Addicted to the Riff

Appalachia can be an unforgiving place to come up in this world. Endemic poverty has coupled with a unique cultural and geographical isolation to create a specific identity that permeates every person, place, and creation within and from the area. This essence is perhaps no better captured than in the music of self-anointed czars of mountain metal, Karma to Burn. Karma to Burn, an instrumental trio based out of Morgantown, West Virginia, has been blazing across the U.S. and European tour trails for more than twenty years and recently released their first official EP, Mountain Czar (out now via...
```

Expected: **Likely Real** · Database · 100%

### 4. FAKE — Kaggle ISOT

```
Man Behind Trump’s Insane Claim That Obama Tapped His ‘Wires’ Just Admitted He Made Sh*t Up (VIDEO)

Donald Trump is now demanding that any probe into his administration s connections to Russia also include a probe into his wild allegation that President Obama tapped his  wires  in Trump Tower. This dubious claim seems to have originated with a Breitbart article that had been circulating among White House staff. The article in question claimed that President Obama had ordered a wiretap under the Foreign intelligence Surveillance Act, which allows a secret court to authorize surveillance in cases that involve agents of foreign powers   in this case, Trump and two Russian banks.The article heav...
```

Expected: **Likely Fake** · Database · 100%

### 5. REAL — All the News 2.1 (15 newsrooms)

```
High performance Ram Rebel TRX on the way with midsize pickup in tow

(Ram) Ram is adding a monster truck and a midsize pickup to its lineup. The automaker confirmed on Friday plans to put its Ram 1500 Rebel TRX concept into production by 2022. The high performance pickup was designed to be a high speed off-roader in the same mold as the Ford F-150 Raptor. 
 (Ram) Details were not revealed, but the 2016 concept featured a 575 hp version of the Dodge Challenger SRT Hellcat’s 6.2-liter supercharged V8, and rumor has it that the production truck will be available with the full-bore 707 hp version. Ram will also reenter the midsize pickup segment that it abandoned w...
```

Expected: **Likely Real** · Database · 100%

### 6. FAKE — Kaggle ISOT

```
WATCH: CNN Host HUMILIATES Trump Supporter For Providing Laughable ‘Evidence’ Of Nationwide Voter Fraud

One of Trump s biggest supporters just got their ass handed to them for having zero evidence of voter fraud.Donald Trump s lies about voter fraud imploded on Monday as CNN tore apart Kris Kobach, the man Stephen Miller cited when asked if he had evidence to prove Trump s claims.Miller repeatedly claimed that millions of Americans voted illegally in November to explain how Trump lost the popular vote to Hillary Clinton. Miller specifically talked about New Hampshire, claiming that thousands of people from Massachusetts bused in to vote a second time in that state. Go to New Hampshire,  Miller s...
```

Expected: **Likely Fake** · Database · 100%

### 7. REAL — All the News 2.1 (15 newsrooms)

```
​Welsh Football Team Not Welsh Enough to be Honoured at Welsh Cultural Festival

The Welsh national football team will not be honoured at this year's National Eisteddfod – the annual festival celebrating the country's cultural heritage – because several players do not speak Welsh. The BBC reports that there had been calls for the team – who reached the semi-finals of Euro 2016 – to be honoured at this year's Eisteddfod, which begins today (July 29) in Abergavenny. But the festival's boss, Archdruid Geraint Lloyd Owen, was having none of it, reasoning that the presence of several non-Welsh-speakers on the team made it impossible to honour them at the event. "If they can't s...
```

Expected: **Likely Real** · Database · 100%

### 8. FAKE — Kaggle ISOT

```
This Compilation Of Alec Baldwin HUMILIATING Trump On SNL Will WRECK Trump New Year’s Eve

Donald Trump threw a party to celebrate the New Year, so let s ruin his good time by looking back at the best of Alec Baldwin s masterful impersonations of him on Saturday Night Live.2016 was a tough year that most Americans would rather forget. We lost beloved musicians, actors, and childhood heroes. But 2016 was a good one for Alec Baldwin as he scored the role of a lifetime because of the stupidity of Donald Trump.Once the presidential debates rolled around in September, Baldwin stepped up and delivered a perfect performance as Trump. It has been so good, in fact, that Trump has whined abou...
```

Expected: **Likely Fake** · Database · 100%

### 9. REAL — Kaggle ISOT

```
Trump considering options for new Afghanistan strategy: White House

WASHINGTON (Reuters) - President Donald Trump is considering his options for a new U.S. strategy in Afghanistan and will make an announcement “at the appropriate time,” White House spokeswoman Sarah Sanders said on Friday after Trump met with his national security team. U.S. officials told Reuters prior to the meeting at Camp David in Maryland that the options being presented to Trump range from a total pullout from Afghanistan, keeping the status quo of some 8,400 U.S. troops, a modest hike, or a small reduction that would focus on counter-terrorism.
```

Expected: **Likely Real** · Database · 100%

### 10. FAKE — McIntire

```
BREAKING: VP Candidate Mike Pence’s Plane Skids Off Runway, Tears Up Tarmac at NYC Airport…

Share on Twitter 
A plane carrying Republican vice presidential candidate Mike Pence skidded off the runway and tore up the tarmac at LaGuardia Airport in New York City on Thursday night. 
The Boeing 737 reportedly “overshot” the runway. There were no reported injuries. 
Pence took to Twitter to ensure the nation that he was unharmed. So thankful everyone on our plane is safe. Grateful for our first responders & the concern & prayers of so many. Back on the trail tomorrow! — Mike Pence (@mike_pence) October 28, 2016 
The videos and photos of the scene were pretty striking. Clearly, the situati...
```

Expected: **Likely Fake** · Database · 100%

---

## Part 2 — Article only, no headline

Paste only the block, no headline on line 1. These now come back as
**Database match · 100%** too, because the app looks the article up by its
opening words as well as by its headline. The result page will show you the
headline it has on file, which is listed under each block so you can check it
picked the right story.

This is what changed: it used to miss here and hand the job to the reading
model. Removing one word from a headline was enough to lose a match that was
sitting right there in the database.

To see the reading model instead, paste something that is genuinely not in the
database — a story published in the last few months.

### 11. REAL — Kaggle ISOT

```
WASHINGTON (Reuters) - Senior U.S. and Chinese officials on Wednesday failed to agree on major steps needed from Beijing to help reduce the U.S. trade deficit with China during an annual economic dialogue meeting, a Trump administration official said. The official, who was not authorized to speak publicly on the talks and requested anonymity, said that the disagreements covered most areas important to the United States, including access to China’s financial services markets, steel overcapacity, trade in autos, Chinese requirements for data localization and ownership caps for foreign firms. But...
```

Truth: **REAL** · 93 words · headline was `U.S., China disagreed on how to reduce U.S. trade deficit: official`

### 12. FAKE — McIntire

```
Home | World | Brexit Lost: Scuppered By May and High Court Brexit Lost: Scuppered By May and High Court By Mr. Charrington 03/11/2016 11:48:44 
LONDON – England – The High Court decision today to deny the EU Referendum result is a sign that we are not living in a democracy or a sovereign country. 

Today is a very sad day for democracy within the UK, and Theresa May a Remain supporter is instrumental in denying the will of the British people to leave the EU. 
Dithering and Delaying Tactics 
Through numerous delaying tactics , and the installation of key government posts of Remain campaign MPs...
```

Truth: **FAKE** · 105 words · headline was `Brexit Lost: Scuppered By May and High Court`

### 13. REAL — Kaggle ISOT

```
WASHINGTON (Reuters) - A Russian oligarch once close to Paul Manafort, President Donald Trump’s former campaign manager, has offered to testify to congressional panels investigating Russian meddling in the 2016 election, but lawmakers are rejecting his conditions, the New York Times reported on Friday, citing congressional officials. The offer by aluminum magnate Oleg Deripaska comes amid growing attention to his ties to Manafort, one of several Trump associates under scrutiny by the Federal Bureau of Investigation over possible collusion with Russia during the presidential campaign. Manafort ...
```

Truth: **REAL** · 87 words · headline was `Russian with ties to former Trump aide wants immunity for testimony: NYT`

### 14. FAKE — McIntire

```
November 11: Daily Contrarian Reads   My daily contrarian reads for Friday, November 11th, 2016. You need to login to view this content. 
David Stockman’s Contra Corner isn’t your typical financial tipsheet. Instead it’s an ongoing dialogue about what’s really happening in the markets… the economy… and governments… so you can understand the world around you and make better decisions for yourself. 
David believes the world -- certainly the United States -- is at a great inflection point in human history. The massive credit inflation of the last three decades has reached its apogee and is now go...
```

Truth: **FAKE** · 101 words · headline was `November 11: Daily Contrarian Reads`

### 15. REAL — McIntire

```
First Read is a morning briefing from Meet the Press and the NBC Political Unit on the day's most important political stories and why they matter.

HEMPSTEAD, NY -- Well, we're finally here: Hillary Clinton and Donald Trump tonight square off in their first presidential debate at Hofstra University, making it arguably the most consequential night so far of the 2016 election. The stakes are enormous, with recent polls showing the national race ranges from a six-point lead for Clinton (in the NBC/WSJ) to a dead-even tie (in Bloomberg's). There are five storylines we're watching heading into the ...
```

Truth: **REAL** · 99 words · headline was `Here are the top five things to watch in tonight's debate`

### 16. FAKE — OpenSources (150 websites)

```
The snack foods that are marketed for children can have anything from cancer causing artificial coloring to the unbelievable, petroleum products! ! Petroleum is the same ingredient that is used to make up oil and gas. Commonly used food dyes, such as Yellow 5, and Red 40 are made from petroleum and pose serious health problems. The Federal Food and Drug Administration (FDA) has stated that ingestion is typically under the “concern threshold.”

Those health problems can include hyperactivity in children, cancer (in animal studies), and allergic reactions. Artificial food coloring has been linke...
```

Truth: **FAKE** · 93 words · headline was `5 Common Children’s Snacks Made with Cancer Causing Petroleum Products`

### 17. REAL — McIntire

```
Washington (CNN) President Barack Obama had some blunt words for Sen. John McCain for questioning the honesty of Secretary of State John Kerry -- telling his former campaign rival to, in essence, back off.

"When I hear some, like Sen. McCain recently, suggest that our secretary of state, John Kerry, who served in the United States Senate, a Vietnam veteran, who's provided exemplary service to this nation, is somehow less trustworthy in the interpretation of what's in a political agreement than the Supreme Leader of Iran -- that's an indication of the degree to which partisanship has crossed a...
```

Truth: **REAL** · 99 words · headline was `Old rivals Obama and McCain tussle over Iran`

### 18. FAKE — OpenSources (150 websites)

```
Demographics today forms a nation’s destiny tomorrow no matter where a man goes. If the people will not have children, then they will die out. This is just a simple fact.

While Christianity has been spreading worldwide, due to the low birth rates of Christians in comparison to the higher rates of Muslims, Christians are dying out and in some cases disappearing completely. It is not just in Europe, but all over the world according to a recent report:

The number of babies born to Muslims is expected to exceed the number born to Christians in under 20 years, according to a new Pew Research repo...
```

Truth: **FAKE** · 107 words · headline was `Muslim Births Outpace Christian Births As Christians Are Dying Out Worldwide`

### 19. REAL — McIntire

```
Donald Trump, Ted Cruz, and John Kasich have all backed away from a pledge to support the Republican presidential nominee. The reasons go deeper than mere personal pique, to the soul of the party.

How SNL's 'the bubble' sketch about polarization is all too true

Republican presidential candidate Donald Trump waves as he walks onstage before speaking at a campaign event at St. Norbert College in De Pere, Wis., on Wednesday, March 30.

When Donald Trump signed a “loyalty pledge” with great fanfare last September promising to support the eventual Republican presidential nominee, few took him ser...
```

Truth: **REAL** · 97 words · headline was `Why the death of GOP 'loyalty pledge' matters`

### 20. FAKE — OpenSources (150 websites)

```
Two events have surreally coincided today in different parts of Damascus.

First there was a fake chemical attack in Eastern Ghouta and shortly thereafter an all too tragic and an all too real treble suicide bombing attack.

Two of the suicide bombers detonated their cars which were loaded with bombs near the Airport Roundabout in Eastern Damascus while a third was able to make it to a downtown area where he set off his bomb.

The death toll is reportedly 19 while over 20 have been cited as injured. According to al-Masdar, the death toll would have been worse had the security services not acte...
```

Truth: **FAKE** · 105 words · headline was `DAMASCUS: 2 attacks–1 real and 1 fake`

---

## Part 3 — See the first-line rule for yourself

Three tests. **Same article all three times.** Only the shape of what you paste
changes. Do them in order, then look at the dashboard.

Before you start: **Clear all** on the dashboard, so you only see these three.

The article is FAKE (Kaggle ISOT).

---

### Test A — headline on line 1, article below it

Paste this whole block:

```
Liberal Group Trolls Trump At Roy Moore Rally In The Best Possible Way (VIDEO)
Donald Trump held a rally for Alabama Senate candidate and alleged pedophile Roy Moore in Pensacola, Florida on Friday night which he later claimed was  packed to the rafters  but the venue was barely half-filled with supporters. Outside of the rally, a liberal group targeted the former reality show star and Moore by using Ivanka Trump s own words.American Bridge used a mobile billboard featuring Ivanka Trump s criticism of Moore.  The truck displayed,  There s a special place in hell for people who prey on children  along with Trump s daughter s picture emblazoned across it.Happening now at @...
```

Expect: **Likely Fake · Database · 100%**

This is the normal case. The app's rule — first line is the title — is correct
here, because you did put the title on the first line.

---

### Test B — same thing, but delete `(VIDEO)` from the headline

Same paste, one word gone off the end of line 1:

```
Liberal Group Trolls Trump At Roy Moore Rally In The Best Possible Way
Donald Trump held a rally for Alabama Senate candidate and alleged pedophile Roy Moore in Pensacola, Florida on Friday night which he later claimed was  packed to the rafters  but the venue was barely half-filled with supporters. Outside of the rally, a liberal group targeted the former reality show star and Moore by using Ivanka Trump s own words.American Bridge used a mobile billboard featuring Ivanka Trump s criticism of Moore.  The truck displayed,  There s a special place in hell for people who prey on children  along with Trump s daughter s picture emblazoned across it.Happening now at @...
```

Expect: **Likely Fake · Database · 100%**

**This is the one you reported.** Before, this came back around 86.9% from the
model, because the headline no longer matched anything. Now the app checks the
article as well, so it still finds it.

Read the "Full explanation" box on the result page. It will tell you the
headline did **not** match and the article did.

---

### Test C — article only, no headline at all

Paste only the article. Nothing above it:

```
Donald Trump held a rally for Alabama Senate candidate and alleged pedophile Roy Moore in Pensacola, Florida on Friday night which he later claimed was  packed to the rafters  but the venue was barely half-filled with supporters. Outside of the rally, a liberal group targeted the former reality show star and Moore by using Ivanka Trump s own words.American Bridge used a mobile billboard featuring Ivanka Trump s criticism of Moore.  The truck displayed,  There s a special place in hell for people who prey on children  along with Trump s daughter s picture emblazoned across it.Happening now at @...
```

Expect: **Likely Fake · Database · 100%**

Here the app's rule is **wrong**. There is no title, so it takes your first
sentence and treats that as the title. It still finds the article, because it
also checks the article's own opening words.

---

### Now go to the dashboard

You should have three rows, all **Likely Fake**, all **Database**, all **100%**.
That is the fix: same answer three times, no matter how you pasted it.

Now look at the **Headline** column. Rows A and B say:

> Liberal Group Trolls Trump At Roy Moore Rally In The Best...

Row C says:

> Donald Trump held a rally for Alabama Senate candidate and...

**That is the leftover I mentioned.** Row C shows your first sentence where a
headline should be, because that is what the app decided the title was. The
verdict is right; the label on the row is just untidy.

The real headline was not lost. Open row C and scroll to **The Article We
Matched** — it is there.

---

## Quick reference

| You paste | Who answers |
| --- | --- |
| Headline only | Database |
| Headline on line 1, article below | Database |
| Article only, no headline | Database |
| Headline with a word changed, article below | Database — matched on the article |
| A story we do not hold, 25 words or more | Reading model |
| Under 25 words and not in the database | Not Sure |
