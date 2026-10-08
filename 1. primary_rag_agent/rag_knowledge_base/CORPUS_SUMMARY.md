# RAG Knowledge Base - Corpus Summary

Generated on: 2025-12-22 05:41:44

## Overview

This corpus contains synthetic enterprise documents for RAG-based reasoning path hijacking research.

## Directory Structure

```
rag_knowledge_base/
├── documents/          # Generated enterprise documents
│   ├── hr_policies/
│   ├── it_security/
│   ├── finance/
│   ├── operations/
│   ├── compliance/
├── metadata/           # Generation metadata (JSON & CSV)
├── generation_plans/   # Document generation plans
└── CORPUS_SUMMARY.md   # This file
```

## Document Categories

### Hr Policies

- Annual Leave and Time-Off Policy (800 words, medium complexity)
- Remote Work and Hybrid Working Policy (1000 words, high complexity)
- Employee Benefits and Perks Handbook (1200 words, high complexity)
- Expense Reimbursement Guidelines (700 words, medium complexity)
- Performance Review and Appraisal Process (900 words, high complexity)

### It Security

- Password and Authentication Policy (600 words, medium complexity)
- Data Classification and Handling Policy (1100 words, high complexity)
- Incident Response and Reporting Procedure (950 words, high complexity)
- Acceptable Use of IT Resources Policy (750 words, medium complexity)
- Cloud Services and SaaS Usage Guidelines (850 words, high complexity)

### Finance

- Procurement and Vendor Management Policy (1000 words, high complexity)
- Budget Planning and Allocation Guidelines (900 words, high complexity)
- Travel and Entertainment Expense Policy (800 words, medium complexity)
- Invoice Processing and Payment Procedures (650 words, medium complexity)

### Operations

- Project Management Methodology and Standards (1100 words, high complexity)
- Internal Communication Protocols (700 words, medium complexity)
- Business Continuity and Disaster Recovery Plan (1200 words, high complexity)
- Document Management and Retention Policy (750 words, medium complexity)

### Compliance

- GDPR and Data Privacy Compliance Guide (1300 words, high complexity)
- Code of Conduct and Ethics Policy (1000 words, high complexity)
- Whistleblowing and Grievance Procedure (800 words, medium complexity)
- Anti-Discrimination and Harassment Policy (950 words, high complexity)
- Intellectual Property and Confidentiality Agreement (900 words, high complexity)

## Next Steps

1. Load documents into RAG system using LangChain
2. Create FAISS vector store
3. Build primary agent with CoT reasoning
4. Prepare for Phase 2: Attack vector implementation
