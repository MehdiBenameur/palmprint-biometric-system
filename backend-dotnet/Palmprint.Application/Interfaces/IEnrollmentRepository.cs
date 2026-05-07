using Palmprint.Domain.Entities;

namespace Palmprint.Application.Interfaces;

public interface IEnrollmentRepository
{
    Task<User?> GetUserByExternalIdAsync(Guid tenantId, string externalId);
    Task AddUserAsync(User user);
    Task AddPalmTemplateAsync(PalmTemplate template);
    Task SaveChangesAsync();
}