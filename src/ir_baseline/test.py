import difflib
import sys
"""

['  Gyil', '  repertoire', '  with', '- master', '  xylophonists', '  Bernard', '  Woma', '- ,', '- Jerome', '- Balsab', '- ,', '- and', '- Alfred', '- Kpebsaane', '  .']
1
['master', '  xylophonists', '  Bernard', '  Woma', ',', 'Jerome', 'Balsab', ',', 'and', 'Alfred', 'Kpebsaane']
['  Gyil', '  repertoire', '  with', '  xylophonists', '+ including', '  Bernard', '  Woma', '  .']
3
['including']



 The so - called Circassian Genocide claimed over " 2 million Circassians " killed by Russia
['  The', '- so', '- -', '- called', '  Circassian', '  Genocide', '- claimed', '  over', '- "', '  2', '  million', '  Circassians', '- "', '  killed', '  by', '  Russia']
['so', '-', 'called', '  Circassian', '  Genocide', 'claimed', '  over', '"', '  2', '  million', '  Circassians', '"']

 The Circassian Genocide over 2 million Circassians killed by Russia
[]
[]



The therapeutic movements may be adapted to the condition of the patient ; for example , they may be executed while sitting or even lying down and have been proven to be helpful for many children ' s ailments .
['  condition', '  of', '  the', '! patient', '  ;', '  for', '  example']
['patient']

The therapeutic movements may be adapted to the condition of the person being treated ; for example , they may be executed while sitting or even lying down .
['  condition', '  of', '  the', '! person', '! being', '! treated', '  ;', '  for', '  example', '***************\n', '*** 24,40 ****\n', '  even', '  lying', '  down', '- and', '- have', '- been', '- proven', '- to', '- be', '- helpful', '- for', '- many', '- children', "- '", '- s', '- ailments', '  .', '--- 26,29 ----\n']
['person', 'being', 'treated', '  ;', '  for', '  example', '***************\n', '*** 24,40 ****\n', '  even', '  lying', '  down', 'and', 'have', 'been', 'proven', 'to', 'be', 'helpful', 'for', 'many', 'children', "'", 's', 'ailments', '  .', '- 26,29 ----\n']




  39   Donald   Cerrone ! detailed ! how   in   recent   years - the - Gym - had - gone - severely - downhill   as   a   result --- 8,23 ----
   39   Donald   Cerrone ! claimed ! Jackson ! Wink ! has ! suffered ! a ! decline   in   recent   years   as   a   result *** 28,31 ****
   '   s   involvement ! . --- 27,38 ----
   '   s   involvement ! , ! calling ! the ! gym ! a ! " ! puppy ! mill ! ."

On the Joe Rogan MMA Show # 39 Donald Cerrone detailed how in recent years the Gym had gone severely downhill as a result of Mike Winklejohn ' s involvement .
['  39', '  Donald', '  Cerrone', '! detailed', '! how', '  in', '  recent', '  years', '- the', '- Gym', '- had', '- gone', '- severely', '- downhill', '  as', '  a', '  result']
['detailed', 'how', '  in', '  recent', '  years', 'the', 'Gym', 'had', 'gone', 'severely', 'downhill']

On the Joe Rogan MMA Show # 39 Donald Cerrone claimed Jackson Wink has suffered a decline in recent years as a result of Mike Winklejohn ' s involvement , calling the gym a " puppy mill ."
['  39', '  Donald', '  Cerrone', '! claimed', '! Jackson', '! Wink', '! has', '! suffered', '! a', '! decline', '  in', '  recent', '  years', '  as', '  a', '  result', '***************\n', '*** 28,31 ****\n', "  '", '  s', '  involvement', '! .', '--- 27,38 ----\n', "  '", '  s', '  involvement', '! ,', '! calling', '! the', '! gym', '! a', '! "', '! puppy', '! mill', '! ."']
['claimed', 'Jackson', 'Wink', 'has', 'suffered', 'a', 'decline', '  in', '  recent', '  years', '  as', '  a', '  result', '***************\n', '*** 28,31 ****\n', "  '", '  s', '  involvement', '.', '- 27,38 ----\n', "  '", '  s', '  involvement', ',', 'calling', 'the', 'gym', 'a', '"', 'puppy', 'mill', '."']




  performance   of   a - raunchy   "   lyrical   battle --- 11,16 ----
 *** 22,25 ****
 --- 21,30 ----
   at   Sting   30 + that + Laing + considered + " + raunchy   . + "

The reason for the ban was due to her impromptu performance of a raunchy " lyrical battle " with Ninja Man at Sting 30 .
['  performance', '  of', '  a', '- raunchy', '  "', '  lyrical', '  battle']
['raunchy']

The reason for the ban was due to her impromptu performance of a " lyrical battle " with Ninja Man at Sting 30 that Laing considered " raunchy . "
['*** 22,25 ****\n', '--- 21,30 ----\n', '  at', '  Sting', '  30', '+ that', '+ Laing', '+ considered', '+ "', '+ raunchy', '  .', '+ "']
['- 21,30 ----\n', '  at', '  Sting', '  30', 'that', 'Laing', 'considered', '"', 'raunchy', '  .', '"']

"""

def extract_diff_v1(pre, post):
    def extract(tok_seq):
        out = []
        island_size = 0 # tracks any "islands" of non-changed words in a changed region
        for tok in tok_seq:
            if tok[0] in '-+!':
                out += [tok[2:]]
                island_size = 0
            else:
                if len(out) > 0:
                    out += [tok]
                island_size += 1

        if island_size:
            return out[:-island_size]

        return out

    diff = difflib.context_diff(pre.split(), post.split())
    # skip header
    
    for _ in range(4): 
        next(diff)

    diff = [x for x in diff if x != '***************\n'] # skip 
    post_start_idx = next( ( (i, x) for i, x in enumerate(diff) if '\n' in x ) )[0]
    
    pre_diff = diff[: post_start_idx]
    post_diff = diff[post_start_idx + 1 :]

    pre_out = extract(pre_diff)
    post_out = extract(post_diff)

    return pre_out, post_out


def extract_diff_v2(pre, post):
    dif_toks = [x for x in difflib.ndiff(pre.split(), post.split())]

    out = []
    cur = None
    for tok in dif_toks:

        if tok.startswith('-'):
            add_idx = 0
        elif tok.startswith('+'):
            add_idx = 1
        elif tok.startswith('?'):
            continue
        else:
            add_idx = -1

        if add_idx < 0 and cur is not None:
            out.append(cur)
            cur = None
        elif add_idx >= 0:
            if cur is None:
                cur = ([], [])
            cur[add_idx].append( tok[2:] )

    if cur is not None:
        out.append(cur)

    return out
    print(dif_toks)
    print(out)

i = 0
for l in open(sys.argv[1]):
    [_, _, pre, post, _] = l.strip().split('\t')

    diffs = extract_diff_v2(pre, post)

    # TODO -- only take simplest case, things get messy w/multiple difs
    if len(diffs) != 1: continue

    i += 1
 #   print(pre)
#    print(post)
#    print(diffs)
    
#    pre_v1, post_v1 = extract_diff_v1(pre, post)
#    print(pre)
#    print(pre_v1)
#    print()
#    print(post)
#    print(post_v1)

#    print('#' * 80)
print(i)
