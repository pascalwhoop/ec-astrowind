---
publishDate: 2024-12-01T00:00:00Z
title: Bootstrapping Enterprise-Grade GCP Infrastructure with Terraform and Terragrunt
excerpt: "Discover how EveryCore leverages Terraform and Terragrunt to build a robust, scalable GCP infrastructure, ensuring efficient management and compliance across all projects."
image: ./cover.jpg
attribution: "Photo by [Anamul Rezwan](https://www.pexels.com/photo/low-angle-photography-of-orange-excavator-under-white-clouds-1078884/)"
category: Technology
tags:
  - technology
  - infrastructure as code
  - terraform
#metadata:
#  canonical: https://astrowind.vercel.app/astrowind-template-in-depth
---

At EveryCore, we've implemented a robust and scalable infrastructure-as-code (IaC) setup using Terraform and Terragrunt to manage our Google Cloud Platform (GCP) infrastructure. Here's an inside look at our approach to cloud infrastructure management.

## Architecture Overview

Our infrastructure follows a hierarchical organization pattern with several key components:

1. **Organization Level Setup** (`gcp-org`)
   - Manages organization-wide policies
   - Configures Security Command Center (SCC) notifications
   - Sets up essential contacts and audit logging
   - Implements budget controls across projects

2. **Bootstrap Layer** (`gcp-bootstrap`)
   - Establishes the foundational infrastructure
   - Sets up initial service accounts and permissions

3. **Resource Organization**
   - Structured folder hierarchy (`03_folders`)
   - Project management (`04_projects`)
   - Centralized policy management (`gcp-policies`)

## Key Features

### 1. Budget Management
We implement sophisticated budget controls across all our projects with:
- Configurable budget thresholds
- Automated alerts at specified spending percentages
- PubSub integration for real-time notifications
- Support for both current and forecasted spend analysis

### 2. Security and Compliance
Our infrastructure emphasizes security through:
- Centralized Security Command Center management
- Role-based access control (RBAC) with Google Workspace integration
- Domain-restricted sharing policies
- Comprehensive audit logging
- Centralized secrets management

### 3. Networking
We support both traditional and hub-and-spoke network architectures with:
- Separate network tiers (base and restricted)
- Configurable interconnect options
- Network policy management
- Granular access controls

### 4. State Management
We utilize:
- Remote state management with GCS backends
- Terragrunt for state file organization
- DRY configuration principles

## Best Practices Implemented

1. **Cost Control**
   - Every project has predefined budget limits
   - Alert thresholds at 120% of budget
   - Separate budgets for different components (networking, secrets, logs)

2. **Security**
   - Data access logging capabilities
   - Security Command Center integration
   - Essential contacts management for security notifications
   - KMS integration for sensitive data

3. **Compliance**
   - Audit logging
   - Domain restrictions
   - Access context management
   - Resource tagging for governance

## Infrastructure as Code Structure

```
infra/deployments/
├── gcp-org/          # Organization-level configurations
├── gcp-bootstrap/    # Initial setup and foundation
├── 03_folders/       # Logical grouping of resources
├── 04_projects/      # Project-specific configurations
└── gcp-policies/     # Centralized policy management
```

This structure allows us to maintain a clear separation of concerns while ensuring consistent policy application across our infrastructure.

## Conclusion

Our infrastructure setup demonstrates how modern cloud infrastructure can be managed at scale using Terraform and Terragrunt. By implementing proper hierarchical organization, security controls, and budget management from the start, we've created a foundation that's both secure and scalable.

The combination of Terraform for resource management and Terragrunt for configuration organization has allowed us to maintain a DRY, maintainable, and secure infrastructure that can grow with our needs while maintaining proper governance and control.
