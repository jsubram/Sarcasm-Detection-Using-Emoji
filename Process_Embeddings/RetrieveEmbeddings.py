import Word2Vec as w
import Emoji2Vec as ex
import Concat as c

def main():
	print("1. Train with Word2Vec, 2. Train with Emoji2Vec 3. Both")
	print("Enter choice (1/2/3):")
	ch = int(input())

	if ch == 1:

		word_vec = w.main()
		return word_vec

	elif ch == 2:
		
		Emoji_vec = ex.main()
		return Emoji_vec

	elif ch == 3:

		print("Concatenating...")
		Concatenated_Vector = c.main()
		return Concatenated_Vector

	else:
		print("Invalid")
