import sys

def main( yaml_file ):
  print "Running: ",yaml_file
  
if __name__ == "__main__":
  #print sys.argv
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_file = sys.argv[1]
  main(yaml_file)