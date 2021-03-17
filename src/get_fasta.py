import urllib.request  # the lib that handles the url stuff

def get_seq(uid):
	target_url = 'https://www.uniprot.org/uniprot/' + uid + '.fasta'
	fasta = ''
	for line in urllib.request.urlopen(target_url):
	    if line.decode('utf-8').replace('\n','')[0] != '>':
	    	fasta += line.decode('utf-8').replace('\n','')


	print(fasta)
	
	return fasta

if __name__ == '__main__':
	uid = 'P32752'
	print(get_fasta(uid))

