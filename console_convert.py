import sys
import re
file_name = sys.argv[1]
matcher = re.compile(r'.*update:(.*) recall:(.*) cost:(.*)\niteration time :(.*)', re.M|re.S) 
with open(file_name) as f:
    strings = f.read()
    lines = strings.split(" seconds\n")
    updates = []
    recalls = []
    times = []
    for i in range(len(lines)-1):
#        print(lines[i])
        m = matcher.match(lines[i])
        updates.append(m.group(1))
        recalls.append(m.group(2))
        times.append(m.group(4))
        
    print(', '.join(updates))
    print(', '.join(recalls))
    print(', '.join(times))

