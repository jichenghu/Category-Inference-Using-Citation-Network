This is a derived from the original Cora Research Paper Classification dataset: 
	http://www.cs.umass.edu/~mccallum/code-data.html 

Dictionary files:
	author_dict.txt, paper_dict.txt, term_dict.txt; format: id+"\t"+name

Relation files:
	PA.txt, PT.txt, PP.txt; format: id1+"\t"+id2+"\t"+frequency
	(PP means id1 cites id2)

Label files:
	paper_label.txt; format id+"\t"+class_label
	(class_label 0 means no label information for this paper)

Mapping relation between areas and their class label:
	CoraHierarchyTree.txt; format area+"\t"+label+"\t"+papers in this area
	(-1 means it is not a leaf area and no label)