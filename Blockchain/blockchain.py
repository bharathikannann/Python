#Libraries
# Flask==0.12.2
# Postman HTTP Client

import datetime
import hashlib
import json
from flask import Flask,jsonify


# Creating a block chain
class Blockchain:
    def __init__(self):
        self.chain=[]
        self.create_block(proof= 1 , previous_hash='0')
    """   
    Creating a block with four fields
    index, timestamp, proof(example used instead of whole block), previous_hash
    """
    def create_block(self, proof, previous_hash):
        block={'index' : len(self.chain) + 1,
                'timestamp' : str(datetime.datetime.now()),
                'proof' : proof,
                'previous_hash' : previous_hash}
        self.chain.append(block)
        return block
    
    # Getting the previous block
    def get_previous_block(self):
        return self.chain[-1]
    
    # Our proof of work for mining the block
    # resultant hash should have 4 leading 0s
    def proof_of_work(self,previous_proof):
        new_proof = 1
        check_proof = False
        
        # create a hash and seeing if new_proof**2 - previous_proof**2 has leading 4 0s else increment the proof and check
        while check_proof is False:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4]=='0000':
                check_proof =  True
            else:
                new_proof +=1
        return new_proof
            
    # Hashing 
    # json. dumps() takes in a json object and returns a string.
    #encode the string to bytes and the hex() method returns a string
    def hash(self,block):
        encoded_block = json.dumps(block, sort_keys = True).encode()
        return hashlib.sha256(encoded_block).hexdigest()
    
    # Checking if a block is valid or not
    def is_chain_valid(self, block):
        previous_block = self.chain[0]
        block_index=1
        while block_index < len(self.chain):
            block = self.chain[block_index]
            
            # check if the prev hash in current block does not match with the original hash of the prev block
            if block['previous_hash'] != self.hash(previous_block):
                return False
            
            # Check if the resultant hash of the proof**2 - previous_proof**2 does not have 4 leading 0s
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(str(proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] !='0000':
                return False
            
            #update the block and increase the index
            previous_block=block
            block_index +=1
        return True
    
    
# Creating a web app
app = Flask(__name__)

# Creating a blockchain
blockchain=Blockchain()

# Mining a block
@app.route('/mine_block', methods = ['GET'])
def mine_block():
    """
    we need previous block and its proof
    calculate the current proof
    create the current block with that proof and prev hash
    """ 
    previous_block = blockchain.get_previous_block()
    previous_proof = previous_block['proof']
    proof = blockchain.proof_of_work(previous_proof)
    previous_hash = blockchain.hash(previous_block)
    block = blockchain.create_block(proof, previous_hash)
    
    # Return the response
    response = {'message' : 'Congrats, you just mined a block!',
                'index' : block['index'],
                'timestamp' : block['timestamp'],
                'proof' : block['proof'],
                'previous_hash' : block['previous_hash']}
    #Response with the JSON representation of the given arguments with an application/json mimetype(Multipurpose Internet Mail Extensions or MIME type).
    return jsonify(response), 200
    
# Getting the blockchain
@app.route('/get_chain', methods = ['GET'])
def get_chain():
    # Return the response
    response = {'chain' : blockchain.chain,
                'length' : len(blockchain.chain)}
    return jsonify(response), 200
    
# Checking if the Blockchain is valid
@app.route('/is_valid', methods = ['GET'])
def is_valid():
    is_valid = blockchain.is_chain_valid(blockchain.chain)
    if is_valid:
        response = {'message': 'All good. The Blockchain is valid.'}
    else:
        response = {'message': 'We have a problem. The Blockchain is not valid.'}
    return jsonify(response), 200


# Running the app
app.run(host = '0.0.0.0', port = 5000)

#http://127.0.0.1:5000/mine_block 
#http://127.0.0.1:5000/get_chain    
"""
hashlib.sha256(str(533**2 - 1**2).encode()).hexdigest()
output - 0000c00870f23a23ae80377298491b091db400d575be0efbde5b310f2f763ed1
Hence the target is acheived and we can create a new block


First 10 blocks
{
  "chain": [
    {
      "index": 1,
      "previous_hash": "0",
      "proof": 1,
      "timestamp": "2020-07-08 17:18:04.659079"
    },
    {
      "index": 2,
      "previous_hash": "2d940288ab55272cb00777512b5099bf647260a333535855e7fc1c89b9dfbea9",
      "proof": 533,
      "timestamp": "2020-07-08 17:18:23.328216"
    },
    {
      "index": 3,
      "previous_hash": "8605f174bc55d1f0cae48d9395b8949f5f74de5e07d7561e85b765dc8aa89534",
      "proof": 45293,
      "timestamp": "2020-07-08 17:18:26.302771"
    },
    {
      "index": 4,
      "previous_hash": "0dd65ce6fc08bca306c3f2e1c8d6b5f6862400123dd112c62824453c5fb5e52d",
      "proof": 21391,
      "timestamp": "2020-07-08 17:18:27.830112"
    },
    {
      "index": 5,
      "previous_hash": "c8d2f17a5d6087f21cf3149b6aeb3a002e55dfc048dee146a0ce99f6abacd68c",
      "proof": 8018,
      "timestamp": "2020-07-08 17:18:29.036412"
    },
    {
      "index": 6,
      "previous_hash": "bec187a4e9bae93c8bb8f4617aae5a438bc974420ad323c2b4628fab225d4505",
      "proof": 48191,
      "timestamp": "2020-07-08 17:18:29.970620"
    },
    {
      "index": 7,
      "previous_hash": "83f7b8115e9bb1ddb307e0f0d9765fffc6d014548bb37d710fd65557deb1ec0c",
      "proof": 19865,
      "timestamp": "2020-07-08 17:18:30.609499"
    },
    {
      "index": 8,
      "previous_hash": "8dd4590d5104d2fe6e7fc48676b729086b1fe40199e788e4c94c2aff7c4e2bd4",
      "proof": 95063,
      "timestamp": "2020-07-08 17:18:31.942466"
    },
    {
      "index": 9,
      "previous_hash": "6458fefb9a2c3631c754f5f712fd35f14d0bf1459168e43663cfef10714752fe",
      "proof": 15457,
      "timestamp": "2020-07-08 17:18:32.398177"
    },
    {
      "index": 10,
      "previous_hash": "7b78bca2e36cb2937185dbcadc05045cd5b9afb980e4df0565e412c0afc7cff2",
      "proof": 15479,
      "timestamp": "2020-07-08 17:27:37.037976"
    }
  ],
  "length": 10
}
"""
    
    
    
    
    
    
    
    
    