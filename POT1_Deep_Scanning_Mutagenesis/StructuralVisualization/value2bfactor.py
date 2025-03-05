from Bio import PDB
import csv

def csv_to_dict(csv_file):
    result_dict = {}

    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            # Assuming the CSV file has a 'residue' column and a 'value' column
            key = row['residue']
            value = row['value']

            # Add the key-value pair to the dictionary, catching untested residues and setting to 1
            try:
                #replaces zeros with 1
                if float(value) == float(0):
                    result_dict[int(key)] = float(default_value)
                else:
                    result_dict[int(key)] = float(value)
            #replaces empty cells with 1
            except ValueError:
                result_dict[int(key)] = float(default_value)

    return result_dict

def replace_b_factor(input_pdb, output_pdb, data_dictionary):
    # Load the PDB structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_pdb)

    # Iterate over all atoms and set the new B factor from dictionary
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    chain_id = chain.id
                    residue_number = residue.id[1]
                    if chain_id == chainToChange: # Specifies a chain
                        bfactor = data_dictionary.get(residue_number,default_value) # Sets the default value to 100.0 if the key is missing
                        atom.set_bfactor(bfactor)
                    else:
                        atom.set_bfactor(default_value)
    # Save the modified structure to a new PDB file
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)

if __name__ == "__main__":
    input_pdb_file = "1xjv.pdb"
    output_pdb_file = "1xjv_del_output.pdb"
    chainToChange = "A"
    default_value=0
    scaling_factor=10
    data_file = "del_data.csv"
    data_dict = csv_to_dict(data_file)


    # Rescale the values to something useful
    data_dict.update((x, y*scaling_factor) for x, y in data_dict.items())
    replace_b_factor(input_pdb_file, output_pdb_file, data_dict)
