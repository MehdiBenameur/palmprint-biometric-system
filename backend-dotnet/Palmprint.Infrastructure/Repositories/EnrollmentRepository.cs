using Microsoft.EntityFrameworkCore;
using Palmprint.Application.Interfaces;
using Palmprint.Domain.Entities;
using Palmprint.Infrastructure.Persistence;

namespace Palmprint.Infrastructure.Repositories;

public class EnrollmentRepository : IEnrollmentRepository
{
    private readonly PalmprintDbContext _context;

    public EnrollmentRepository(PalmprintDbContext context)
    {
        _context = context;
    }

    public async Task<User?> GetUserByExternalIdAsync(Guid tenantId, string externalId)
    {
        return await _context.Users
            .FirstOrDefaultAsync(x => x.TenantId == tenantId && x.ExternalId == externalId);
    }

    public async Task AddUserAsync(User user)
    {
        await _context.Users.AddAsync(user);
    }

    public async Task AddPalmTemplateAsync(PalmTemplate template)
    {
        await _context.PalmTemplates.AddAsync(template);
    }

    public async Task SaveChangesAsync()
    {
        await _context.SaveChangesAsync();
    }
}