using Palmprint.Application.DTOs;
using Palmprint.Application.Interfaces;
using Palmprint.Domain.Entities;

namespace Palmprint.Application.Services;

public class EnrollmentService : IEnrollmentService
{
    private readonly IEnrollmentRepository _repository;
    private readonly IAiServiceClient _aiServiceClient;
    private readonly ITemplateSecurityService _templateSecurityService;
    private readonly IOperationLogger _operationLogger;

    public EnrollmentService(
        IEnrollmentRepository repository,
        IAiServiceClient aiServiceClient,
        ITemplateSecurityService templateSecurityService,
        IOperationLogger operationLogger)
    {
        _repository = repository;
        _aiServiceClient = aiServiceClient;
        _templateSecurityService = templateSecurityService;
        _operationLogger = operationLogger;
    }

    public async Task<EnrollResponse> EnrollAsync(EnrollRequest request)
    {
        try
        {
            if (request.TenantId == Guid.Empty)
                throw new ArgumentException("TenantId is required.");

            if (string.IsNullOrWhiteSpace(request.FullName))
                throw new ArgumentException("FullName is required.");

            if (string.IsNullOrWhiteSpace(request.ExternalId))
                throw new ArgumentException("ExternalId is required.");

            if (request.ImageBytes.Length == 0)
                throw new ArgumentException("Image is required.");

            var user = await _repository.GetUserByExternalIdAsync(request.TenantId, request.ExternalId);

            if (user is null)
            {
                user = new User
                {
                    TenantId = request.TenantId,
                    FullName = request.FullName,
                    ExternalId = request.ExternalId
                };

                await _repository.AddUserAsync(user);
            }

            var aiResult = await _aiServiceClient.GenerateEmbeddingAsync(request.ImageBytes);

            var encryptedEmbedding = _templateSecurityService.EncryptEmbedding(aiResult.Embedding);
            var embeddingHash = _templateSecurityService.HashEmbedding(aiResult.Embedding);

            var template = new PalmTemplate
            {
                TenantId = request.TenantId,
                UserId = user.Id,
                EncryptedEmbedding = encryptedEmbedding,
                EmbeddingHash = embeddingHash,
                ModelVersion = aiResult.ModelVersion,
                TemplateVersion = "v1",
                QualityScore = aiResult.QualityScore,
                IsActive = true
            };

            await _repository.AddPalmTemplateAsync(template);
            await _repository.SaveChangesAsync();

            await _operationLogger.LogAsync(
                request.TenantId,
                user.Id,
                "ENROLLMENT",
                true,
                null,
                "Enrollment completed successfully.",
                null
            );

            return new EnrollResponse
            {
                UserId = user.Id,
                TemplateId = template.Id,
                QualityScore = aiResult.QualityScore,
                Message = "Enrollment completed successfully."
            };
        }
        catch (Exception ex)
        {
            await _operationLogger.LogAsync(
                request.TenantId,
                null,
                "ENROLLMENT",
                false,
                null,
                ex.Message,
                null
            );

            throw;
        }
    }
}