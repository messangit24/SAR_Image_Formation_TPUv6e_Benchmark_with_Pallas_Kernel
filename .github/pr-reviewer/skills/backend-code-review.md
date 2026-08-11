---
name: backend-code-review
description: Reviews backend Java code for boilerplate removal, conventions, and completeness. Use when user asks to "review backend code", "code review java", or "/backend-code-review".
phase: review
user-invocable: true
---

# Backend Java Code Review

Perform a focused code review for backend Java code in the project.

## Scope

**IMPORTANT**: This review should ONLY examine code in the Java services workspace under `app-services/java-services/`.
Do not review code in other directories (frontend, scripts, etc.).

When invoked, prefer glob patterns like:
- `app-services/java-services/**/src/main/java/**/*.java`
- `app-services/java-services/**/src/test/java/**/*.java`

**CRITICAL**: Verify all code uses the shared libraries and patterns established for the project:
- **api-commons**: Exception handling with BusinessException/TechnicalException and ErrorCode
- **logging**: DLP-aware logging with DlpLogger (NOT slf4j Logger directly)
- **redaction**: PII redaction (transitively through logging)
- **mock-data**: Test data generation with MockDataGenerator and MockDataStore


## Primary Focus Areas

### 1. Boilerplate Code Removal
Check for and flag unnecessary boilerplate:
- **Lombok Opportunities**: Replace getters/setters/constructors with @Data, @Builder, @AllArgsConstructor, @NoArgsConstructor
- **Redundant Annotations**: Remove duplicate or unnecessary Spring annotations
- **Verbose Null Checks**: Replace with Optional, Objects.requireNonNull, or @NonNull
- **Unnecessary Comments**: Remove obvious comments that restate the code
- **Empty Catch Blocks**: Flag or remove empty exception handlers
- **Unused Imports**: Flag unused imports
- **Unnecessary Access Modifiers**: Remove redundant public modifiers in interfaces
- **Verbose Stream Operations**: Simplify overly complex stream chains
- **Default Constructor Boilerplate**: Remove if not needed or use Lombok
- **ToString/Equals/HashCode**: Replace manual implementations with @EqualsAndHashCode, @ToString

### 2. Completeness Check ("Required Stuff")
Verify all required elements are addressed:
- **TODO/FIXME Comments**: Flag any unresolved TODOs or FIXMEs
- **Required Validations**: Check @Valid, @NotNull, @NotBlank are present where needed
- **Exception Handling**: Verify proper exception handling with BusinessException/TechnicalException
- **GlobalExceptionHandler**: Check service has GlobalExceptionHandler with @RestControllerAdvice
- **Transaction Management**: Confirm @Transactional where needed
- **Security Annotations**: Check @PreAuthorize, @Secured for protected endpoints
- **API Documentation**: Verify @Operation, @ApiResponse on REST endpoints (OpenAPI 3.0)
- **DLP-Aware Logging**: Confirm DlpLogger is used (NOT slf4j Logger directly)
- **Test Coverage**: Flag missing unit tests for business logic
- **Null Safety**: Check for potential NullPointerExceptions
- **Resource Cleanup**: Verify try-with-resources or proper close() calls

### 3. Java/Spring Boot Best Practices
- **Dependency Injection**: Constructor injection with @RequiredArgsConstructor (not @Autowired fields)
- **Immutability**: Use final fields where possible
- **DTOs vs Entities**: Proper separation, never expose entities directly
- **Repository Layer**: No business logic in repositories (or DAOs with JdbcTemplate)
- **Service Layer**: Clean separation of concerns, throw BusinessException for errors
- **REST Controllers**: Thin controllers, delegate to services, no try-catch blocks
- **Exception Handling**: GlobalExceptionHandler with @RestControllerAdvice (or reactive version for WebFlux)
- **Validation**: Bean Validation annotations on DTOs (@NotNull, @NotBlank, @Valid)

### 4. Shared Library Patterns

#### Exception Handling (api-commons)
- **NEVER use RuntimeException**: Use BusinessException (4xx) or TechnicalException (5xx)
- **Use ErrorCode enum**: All exceptions must use ErrorCode (e.g., ErrorCode.CASE_NOT_FOUND)
- **Add context**: Use .addContext(key, value) to add debugging information
- **No try-catch in controllers**: Let GlobalExceptionHandler catch and convert to ApiError
- **Service methods throw exceptions**: Services throw BusinessException, controllers don't catch

#### Logging (shared/java/logging)
- **Use DlpLogger**: Import `com.generic.shared.logging.DlpLogger`
- **Use DlpLoggerFactory**: DlpLoggerFactory.getLogger(ClassName.class)
- **NEVER use @Slf4j**: This imports slf4j Logger directly, bypassing DLP redaction
- **NEVER import slf4j**: No import org.slf4j.Logger or org.slf4j.LoggerFactory
- **PII auto-redacted**: SSNs, emails, phones automatically redacted in logs

#### Test Data (shared/java/mock-data)
- **Use MockDataGenerator**: For realistic test data (names, addresses, SSNs, etc.)
- **Use MockDataStore**: For in-memory placeholder data during development
- **Custom-specific generators**: `MockDataGenerator.customAddress()`, `customCounty()`, etc.

#### Project-Specific
- **PII Protection**: DlpLogger handles automatic redaction
- **Audit Trail**: Ensure audit events fired for state changes
- **Microservice Boundaries**: Respect domain service boundaries
- **Event-Driven**: Check pub/sub patterns for cross-workflow coordination

## Review Process

1. **Read all files** being reviewed thoroughly
2. **Check shared library usage** - verify DlpLogger, BusinessException, ErrorCode patterns
3. **Scan for anti-patterns** - list every violation with line numbers
4. **Check for TODOs** - flag any unresolved items
5. **Verify completeness** - confirm all validations, exception handling, security present
6. **Flag issues** by severity
7. **Save the report** — write output to `.claude/scratch/<ticket>-<slug>/backend-code-review.md`. Derive `<ticket>` from the branch name (e.g., `feat/1234-feature` → `1234`); derive `<slug>` from the service or feature name. Omit `<ticket>-` prefix if no ticket is found. Example: `.claude/scratch/1234-case-service/backend-code-review.md`.

### Priority Checks (In Order)
1. ⚠️ **CRITICAL**: slf4j Logger usage instead of DlpLogger (PII leak risk)
2. ⚠️ **CRITICAL**: RuntimeException instead of BusinessException/TechnicalException
3. ⚠️ **CRITICAL**: Missing GlobalExceptionHandler in service
4. 🟡 **Important**: @Slf4j annotation usage
5. 🟡 **Important**: Try-catch blocks in controllers
6. 🟡 **Important**: Missing ErrorCode usage
7. 🟢 **Suggestion**: Boilerplate code (manual getters/setters, etc.)

## Output Format

```markdown
## Java Code Review: [Filename]

### Summary
[1-2 sentence assessment]

### Boilerplate to Remove
🧹 **[Line X-Y]**: [Description]
   - Before: [Code snippet]
   - After: [Simplified version]

### Completeness Issues
⚠️ **[Line X]**: [Missing item]
   - Required: [What's needed]
   - Fix: [How to address]

### Issues by Severity
#### 🔴 Critical
- **[Line X]**: [Security/breaking issue]

#### 🟡 Important
- **[Line X]**: [Code quality concern]

#### 🟢 Suggestions
- **[Line X]**: [Nice-to-have improvement]

### Strengths
- [What's done well]

### Action Items
1. [Remove boilerplate at lines X-Y]
2. [Add missing validation at line Z]
3. [Address TODO at line W]
```

## Common Anti-Patterns to Flag

### Boilerplate Code
```java
// BAD - Verbose getters/setters
public String getName() { return name; }
public void setName(String name) { this.name = name; }
// GOOD - Use @Data or @Getter/@Setter

// BAD - Manual null check
if (object != null) { object.doSomething(); }
// GOOD - Optional.ofNullable(object).ifPresent(Object::doSomething)

// BAD - Field injection
@Autowired private final MyService service;
// GOOD - Constructor injection with @RequiredArgsConstructor
private final MyService service;

// BAD - Exposing entities
@GetMapping("/users") List<User> getUsers() { return userRepo.findAll(); }
// GOOD - Return DTOs, map in service layer
```

### Exception Handling (CRITICAL)
```java
// ❌ WRONG - Using RuntimeException
throw new RuntimeException("Case not found: " + id);

// ✅ CORRECT - Using BusinessException with ErrorCode
throw new BusinessException(ErrorCode.CASE_NOT_FOUND, "Case not found: " + id)
    .addContext("caseId", id);

// ❌ WRONG - Try-catch in controller
@GetMapping("/{id}")
public ResponseEntity<CaseDTO> getCase(@PathVariable Long id) {
    try {
        return ResponseEntity.ok(service.getCase(id));
    } catch (Exception e) {
        return ResponseEntity.notFound().build();
    }
}

// ✅ CORRECT - Let GlobalExceptionHandler handle it
@GetMapping("/{id}")
public ResponseEntity<CaseDTO> getCase(@PathVariable Long id) {
    return ResponseEntity.ok(service.getCase(id));
    // Service throws BusinessException, GlobalExceptionHandler converts to ApiError
}

// ❌ WRONG - Empty or generic catch
try { ... } catch (Exception e) { }

// ✅ CORRECT - Throw BusinessException or TechnicalException
try {
    externalService.call();
} catch (IOException e) {
    throw new TechnicalException(ErrorCode.EXTERNAL_SYSTEM_ERROR,
        "Failed to connect to external service", e);
}
```

### Logging (CRITICAL)
```java
// ❌ WRONG - Using @Slf4j (imports slf4j directly)
@Slf4j
@Service
public class CaseService {
    // This bypasses DLP redaction!
}

// ❌ WRONG - Importing slf4j Logger
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
private static final Logger log = LoggerFactory.getLogger(CaseService.class);

// ✅ CORRECT - Using DlpLogger
import com.generic.shared.logging.DlpLogger;
import com.generic.shared.logging.DlpLoggerFactory;
private static final DlpLogger log = DlpLoggerFactory.getLogger(CaseService.class);

// ✅ CORRECT - PII automatically redacted
log.info("Processing case for SSN: 123-45-6789");
// Logs: "Processing case for SSN: [REDACTED_SSN]"
```

### Test Data Generation
```java
// ❌ WRONG - Hardcoded test data
String name = "John Doe";
String email = "john@example.com";
String ssn = "123-45-6789";

// ✅ CORRECT - Using MockDataGenerator
import com.generic.shared.mock.MockDataGenerator;
String name = MockDataGenerator.fullName();
String email = MockDataGenerator.email();
String ssn = MockDataGenerator.ssn();
String address = MockDataGenerator.customAddress();
```

## After Review

Ask the user:
1. "Would you like me to remove the boilerplate code I found?"
2. "Should I fix the completeness issues?"
3. "Would you like a detailed refactoring plan?"
```
