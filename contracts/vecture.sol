// SPDX-License-Identifier: MIT
//  vecture 0.0.1
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract vecture_01 is ERC721, Ownable {
    struct Meta {
        bytes32 contentHash;   // kanonik metnin hash'i (utf8_nfkc_lower_strip_html)
        bytes32 vectorHash;    // embedding vector'ün hash'i (semantic duplicate engelle)
        string  manifestCID;   // ipfs://... (manifest içinde embeddings URI vs)
        string  sourceUrl;     // kaynak URL (https://...)
    }

    uint256 public nextId = 1;
    mapping(uint256 => Meta) private _metaById;
    mapping(bytes32 => bool) public usedContentHash; // duplicate engelle
    mapping(bytes32 => bool) public usedVectorHash;  // semantic duplicate engelle

    event Minted(uint256 indexed tokenId, bytes32 indexed contentHash, bytes32 indexed vectorHash, string manifestCID, string sourceUrl);

    constructor() ERC721("VectorRecord", "VEC") Ownable(msg.sender) {}

    function mint(bytes32 contentHash, bytes32 vectorHash, string calldata manifestCID, string calldata sourceUrl) external returns (uint256 tokenId) {
        require(!usedContentHash[contentHash], "duplicate content");
        require(!usedVectorHash[vectorHash], "duplicate vector");
        tokenId = nextId++;
        _safeMint(msg.sender, tokenId);
        _metaById[tokenId] = Meta(contentHash, vectorHash, manifestCID, sourceUrl);
        usedContentHash[contentHash] = true;
        usedVectorHash[vectorHash] = true;
        emit Minted(tokenId, contentHash, vectorHash, manifestCID, sourceUrl);
    }

    function getMeta(uint256 tokenId) external view returns (Meta memory) {
        require(_ownerOf(tokenId) != address(0), "nonexistent");
        return _metaById[tokenId];
    }
}