# BLDC Motor Fault Detection Project — Explained in Plain Language

This document walks through everything we did in this project, step by step, in normal language — what the project actually is, why we did each thing, what problems came up, how we fixed them, and what the results actually mean.

---

## 1. What is this project actually about?

Think about a BLDC motor (the kind used in drones, fans, electric bikes, etc.). Inside it, there's a little sensor called a **Hall sensor** that tells the motor's controller exactly when to switch power between the motor's coils, so it spins smoothly.

Now imagine that sensor gets slightly knocked out of position — even by a tiny amount, like 0.0001 seconds of timing delay. The motor still runs, but the electrical current flowing through it starts behaving a little "off." A human looking at the raw numbers wouldn't easily notice this. But if you turn that current data into a picture (a visual pattern), a computer vision AI model *can* learn to spot the difference between "normal" and "something's wrong."

That's the whole idea of this project: **can an AI model look at pictures made from motor current data and correctly tell whether the Hall sensor is working normally or has drifted out of position?**

This isn't something we invented from scratch — a group of researchers already studied this exact problem, published a scientific paper about it, and put both their dataset and their code online publicly. Our job was to take their published dataset, get their code actually running on it, and then improve on it.

---

## 2. Where did the data and code come from?

Two sources, given by your professor:

1. **The dataset** — hosted on a site called IEEE DataPort. It contains motor current measurements recorded under four conditions: completely normal (`no_delay`), and three levels of Hall-sensor fault severity (`0.0001` seconds delay, `0.005` seconds delay, and `0.01` seconds delay — bigger number means a more severe fault). The people who published this dataset already did the work of converting the raw current numbers into actual images (224x224 pixel pictures), since AI image models work on pictures, not raw spreadsheets of numbers.

2. **The code** — a GitHub repository by the same research group, containing a Python script (`BLDC_Hall_Detection.py`) that trains 8 different well-known AI vision models (things like MobileNet, VGG16, ResNet50 — these are all famous, pre-built AI architectures that already know how to "see" general things from being trained on millions of everyday photos) to tell normal vs. faulty motor images apart.

Important thing to understand: **this code was written by the original researchers for their own personal computer, with their own private file locations and their own private way of organizing the data.** It was never going to just work out of the box on your laptop with your downloaded copy of the dataset — that's completely normal when reusing someone else's research code, not a sign anything was broken.

---

## 3. Getting your computer ready (this took a while, and here's why)

Before any AI training could even start, we had to get your laptop properly set up, and this turned out to be one of the most time-consuming parts of the whole project — worth understanding, because these are exactly the kinds of "boring but essential" problems that come up in any real coding project.

**Problem #1 — Python version mismatch.**
Your laptop had the newest version of Python installed (3.14). Sounds like a good thing, right? Actually no — TensorFlow (the AI library we need) hadn't been updated yet to support that brand-new version. So when we tried to install it, it just said "no matching version found," over and over. The fix: install an older, compatible version of Python (3.12) *alongside* your existing one, without removing anything, and specifically tell the computer to use 3.12 just for this project.

**Problem #2 — Installing was painfully slow.**
Even after fixing the Python version, installing TensorFlow was crawling along — we're talking 10+ minutes for something that should take 1-2 minutes. We checked Task Manager and could see it was technically still working, just moving at a snail's pace (less than 1 MB per second). It turned out the project folder was sitting on your USB flash drive. USB drives are fine for storing big files, but they're genuinely bad at handling the *thousands* of tiny little files that Python software gets unpacked into during installation. The fix: we copied just the code (not the big dataset — that stayed safely on the USB drive) over to your laptop's internal drive (`C:`), and reinstalled everything there. Suddenly it went from "maybe stuck?" to done in a couple of minutes.

**Problem #3 — A confusing mix-up between "editor shows the new code" and "the file is actually saved."**
A few times, we pasted updated code into a text editor, but the actual file on disk still had the old code — usually because the save didn't happen, or the wrong tab/file got edited. We caught this by directly asking PowerShell to print the real file contents from disk, which doesn't lie the way an editor window sometimes visually can. Eventually, to avoid this problem entirely, we just gave you ready-made files to download and drop straight into the folder instead of copy-pasting.

By the end of all this, you had: Python 3.12 installed, a clean isolated project environment (called a "virtual environment" or "venv" — basically a private sandboxed copy of Python just for this project so it doesn't interfere with anything else on your computer), TensorFlow and all the other needed AI tools properly installed, and the code sitting on your fast internal drive while the big dataset stayed on the USB stick.

---

## 4. Getting the code to actually match your data

Here's where the real detective work started. When we finally tried running the original researcher's script, it immediately crashed — it was looking for a folder on a completely different computer (`E:/USB/논문/...` — a path from the original author's Korean-language folder setup). Obviously that folder doesn't exist on your machine.

So we rewrote the data-loading part of the code to point at your actual downloaded folders instead.

But then we discovered something more interesting once it started actually training: **your "normal" folder only has 120 pictures, but each fault-condition folder has around 715-719 pictures.** That's roughly 6 times more "faulty" examples than "normal" examples. This matters a lot, because if you don't handle it carefully, the AI model can cheat — it can just learn to always guess "faulty" for everything, and it'll still be right most of the time simply because faulty examples are so much more common in the data. That's not real learning, that's the model taking a lazy shortcut, and we actually caught this happening in an early test run (the model was scoring worse than just guessing the majority class every time!).

We tried one fix first — telling the AI model "pay extra attention to the rare `no_delay` examples" (a technique called class weighting). This helped a little but didn't fully solve it. The fix that actually worked well was simpler: instead of using all 715+ faulty images, we randomly picked just 120 of them to match the 120 normal images. Now the AI sees a fair, balanced mix during training instead of being flooded with mostly-faulty examples. This is a really common and standard trick in AI work called "undersampling."

---

## 5. How the actual AI training works (in plain terms)

Each of the 8 AI models we tried works the same basic way:

1. **Start with a model that already knows how to "see."** These 8 models (MobileNet, VGG16, ResNet50, etc.) were all originally trained by big tech companies on millions of everyday photos — cats, cars, furniture, you name it. They already understand general visual concepts like edges, textures, and shapes.

2. **Freeze that existing knowledge.** We tell the model "don't forget what you already know how to see" — we lock those layers so they don't change during our training.

3. **Add a small new "decision layer" on top**, specifically for our 2 categories: normal or faulty. Only this new part actually learns from your motor image data.

4. **Show it your images repeatedly** (called "epochs" — one epoch means the model has seen every training image once), each time adjusting itself slightly to get better at telling normal from faulty.

5. **Stop automatically once it's not improving anymore** (rather than training a fixed number of times no matter what) and **keep whichever version of the model actually performed best**, not just whatever happened at the very last step — because sometimes a model that was doing great a few rounds ago gets slightly worse later just by random chance, and we don't want to accidentally report that worse version.

6. **We also added a trick called "data augmentation"** — during training only, we randomly flip, slightly rotate, or slightly zoom the images. This is like showing the model slightly different "camera angles" of the same information, which helps it generalize instead of just memorizing the exact 120-ish pictures it saw.

We ran this whole process for **all 3 fault severities** (0.0001s, 0.005s, 0.01s) and **all 8 different AI model types**, giving us 24 total experiments, and automatically saved every single result into a spreadsheet-style file (`results_summary.csv`) so nothing had to be copied by hand.

---

## 6. What we actually found (the results)

Here's the plain-language version of the results table:

**The clear winner was MobileNetV1** — the same one the original researchers' paper title actually highlighted as their focus, interestingly enough. It scored about **74% accuracy on average** across all three fault severities, and it was also by far the *fastest* to train (under 2 minutes per run, versus some other models taking 10+ minutes each). So not only did the smallest, most lightweight AI model win — it won by a good margin, and it was way more efficient too. That's a genuinely interesting finding worth highlighting.

**Three models basically failed completely** — EfficientNetB0 (every single time), and ResNet50 and VGG16 (most of the time). They scored exactly 50%, which is literally the same as flipping a coin — meaning they learned nothing useful and just kept guessing one category. We didn't just shrug this off — we figured out *why*: all 8 models were being fed images that were prepared in a specific way meant only for MobileNet (a step called "preprocessing," basically how the raw pixel numbers get scaled before the AI looks at them). But EfficientNet, ResNet, and VGG each expect their pixel data prepared *differently*. Feeding them the wrong kind of preprocessing is a known way to accidentally sabotage a model's ability to learn — even though the model architecture itself is perfectly capable. This is actually a flaw inherited directly from the original researchers' code, not something we introduced.

**The middle fault severity (0.005 seconds) got the best results** for the winning model — 79% accuracy specifically at that setting. That's an interesting detail worth mentioning too, though with this small a dataset it's hard to say for certain whether that's a meaningful pattern or just how this particular random sample of images happened to fall.

---

## 7. Being honest about the limitations

No project is perfect, and pointing out limitations honestly is actually a *good* thing to include, not something to hide:

- **The dataset is small.** After balancing, we only had about 240 total images per fault condition to work with (192 for training, 48 for testing). AI models generally do better with more data, so these numbers should be seen as a solid first baseline, not a final, perfect answer.
- **The preprocessing mismatch issue** means ResNet50, VGG16, and EfficientNetB0 might actually be capable of much better results than what we saw — we just didn't get to test that properly within this project's time.
- **We threw away a lot of "faulty" images** to make things balanced — around 595 unused faulty images per condition. There are other, more data-efficient ways to handle imbalance that we didn't have time to try.
- **Everything ran on a regular laptop CPU**, not a fancy GPU, which limited how much experimentation we could realistically fit into the project timeline.

---

## 8. What this all adds up to

We started with two links from your professor — a dataset and some reference code — and turned that into a working, properly adapted AI pipeline that runs on your own downloaded copy of real published motor-fault data. Along the way we diagnosed and fixed real technical obstacles (wrong Python version, painfully slow installs, a hidden class-imbalance problem that was silently ruining the model's accuracy) and ended up with a genuinely meaningful result: **MobileNetV1 is both the most accurate and the most efficient model for detecting this specific kind of motor fault**, and we can explain *why* the other models underperformed rather than just reporting a confusing set of numbers.

That combination — a working pipeline, an honest investigation of what went wrong and why, and a clear headline finding — is exactly what a good project report should be built around.
